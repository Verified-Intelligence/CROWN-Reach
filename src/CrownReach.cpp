#include "CrownReach.h"
#include <fstream>

vector<Result_of_Reachability> CrownReach(vector<Flowpipe>& initial_sets,
                                          ODE<Real>& dynamics, 
                                          Computational_Setting& setting, 
                                          Variables& vars, 
                                          vector<unsigned int>& sys_settings,
                                          unsigned int steps,
                                          double step_size,
                                          int* u_ids, 
                                          vector<vector<Interval>>* initial_boxes,
                                          vector<string>* constraints_safe,
                                          vector<string>* constraints_unsafe,
                                          vector<string>* constraints_target,
                                          bool test_mode){
    // read nn_settings
    unsigned int numVars = sys_settings[0];
    unsigned int num_nn_input = sys_settings[1];
    unsigned int num_nn_output = sys_settings[2];


    // RPC client
    HttpClient httpclient("http://127.0.0.1:5000");
    Client cl(httpclient, JSONRPC_CLIENT_V2);


    vector<Constraint> safeSet;
    if (constraints_safe!=nullptr){
        for (int i = 0; i < constraints_safe->size(); i++)
        {
            Constraint c_temp((*constraints_safe)[i], vars);
            safeSet.push_back(c_temp);
        }
    }

    vector<Symbolic_Remainder> symbolic_remainders;
    vector<Result_of_Reachability> results;
    for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
    {
        Flowpipe initial_set = initial_sets[sub_iter];
        Symbolic_Remainder symbolic_remainder_temp(initial_set, 1000);
        symbolic_remainders.push_back(symbolic_remainder_temp);
        Result_of_Reachability result_temp;
        results.push_back(result_temp);
    }

    auto start = std::chrono::steady_clock::now();
    unsigned int num_threads = boost::thread::hardware_concurrency();

    // test mode setting
    unsigned int max_steps = test_mode ? std::min(steps, 5U) : steps;
    std::ofstream resultsFile;
    if (test_mode) {
        std::string resultsFileName = "../tests/test_epoch_results.txt";
        resultsFile.open(resultsFileName, std::ios_base::out); 
    }

    for (int iter = 0; iter < max_steps; iter++)
    {
        boost::asio::thread_pool pool(num_threads);
        cout << "Step " << iter << endl;
        
        Json::Value input_lb(Json::arrayValue);
        Json::Value input_ub(Json::arrayValue);
        for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
        {
            for (int i = 0; i < num_nn_input; i++)
            {
                Interval input_range_temp;
                initial_sets[sub_iter].tmvPre.tms[i].intEval(input_range_temp, initial_sets[sub_iter].domain);
                input_lb.append(input_range_temp.inf());
                input_ub.append(input_range_temp.sup());
            }
        }

        // Call CROWN
        Json::Value params, output_coefficients;
        params["input_lb"] = input_lb;
        params["input_ub"] = input_ub;
        cl.CallMethod("CROWN_reach", params, output_coefficients);
        // cout << "Received output coefficients: " << params["input_lb"][0] <<params["input_ub"][0] << endl;
        // cout << "Received output coefficients: " << output_coefficients.toStyledString() << endl;
        if (test_mode) {
            resultsFile << "Epoch " << iter << " - Received output lower/upper bounds: " << params.toStyledString() << std::endl;
            resultsFile << "Epoch " << iter << " - Received output coefficients: " << output_coefficients.toStyledString() << std::endl;
        }
        // Unpack results from CROWN
        for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
        {
            Matrix<Real> T(num_nn_output, num_nn_input, Real(0));
            vector<Real> c_vector;
            vector<double> interval_r;
            for (int j = 0; j < num_nn_output; j++)
            {
                for (int i = 0; i < num_nn_input; i++)
                {
                    T[j][i] = output_coefficients["T"][sub_iter][j][i].asFloat();
                }
                double u_max = output_coefficients["u_max"][sub_iter][j].asFloat();
                double u_min = output_coefficients["u_min"][sub_iter][j].asFloat();
                c_vector.push_back((u_max + u_min) / 2);
                interval_r.push_back((u_max - u_min) / 2);
            }
            
            // Construct new Taylor Models
            TaylorModelVec<Real> tmv_output(c_vector, numVars + 1);
            for (int j = 0; j < num_nn_output; j++)
            {
                for (int i = 0; i < num_nn_input; i++)
                {
                    tmv_output.tms[j] += initial_sets[sub_iter].tmvPre.tms[i] * T[j][i];
                }
                Interval remainder_temp(-interval_r[j], interval_r[j]);
                tmv_output.tms[j].remainder += remainder_temp;
            }

            for (int j = 0; j < num_nn_output; j++)
            {
                initial_sets[sub_iter].tmvPre.tms[u_ids[j]] = tmv_output.tms[j];
            }
        }

        // Flow*
        std::atomic<bool> terminate(false);
        int in_safeset = 0;
        for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
        {
            boost::asio::post(pool, [&, sub_iter] {
                // excute task
                dynamics.reach(results[sub_iter], initial_sets[sub_iter], step_size, setting, safeSet, symbolic_remainders[sub_iter]);

                // if in safeset
                if (constraints_safe != nullptr){
                    if (results[sub_iter].status == COMPLETED_UNSAFE)
                    {
                        cout << "Unsafe." << endl;
                        in_safeset = 1;
                    }
                    else if (results[sub_iter].status == COMPLETED_UNKNOWN)
                    {
                        cout << "Unknown." << endl;
                        in_safeset = 2;
                    }
                }

                if (results[sub_iter].status == COMPLETED_SAFE || results[sub_iter].status == COMPLETED_UNSAFE || results[sub_iter].status == COMPLETED_UNKNOWN)
                {
                    initial_sets[sub_iter] = results[sub_iter].fp_end_of_time;
                }
                else
                {
                    cout << "Flow* terminated." << endl;
                    cout << "Broken branch: " << sub_iter << endl;
                    if (initial_boxes != nullptr){
                        cout << (*initial_boxes)[sub_iter][0] << "\t" << (*initial_boxes)[sub_iter][1] << endl;
                    }
                    terminate.store(true);
                }
            });
        }
        pool.join();
        if (terminate.load() || in_safeset!=0)
        {
            break;
        }
    }

    // Check obstacle
    int safe = 0;
    if (constraints_unsafe != nullptr){
        safe = 0;
        vector<Constraint> unsafeSet;
        for (int i = 0; i < constraints_unsafe->size(); i++)
        {
            Constraint c_temp((*constraints_unsafe)[i], vars);
            unsafeSet.push_back(c_temp);
        }
        for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
        {
            results[sub_iter].unsafetyChecking(unsafeSet, setting.tm_setting, setting.g_setting);
            if (results[sub_iter].isSafe() && constraints_target == nullptr)
            {
                safe = 2;   // Safe
                cout << "Verified" << endl;
                break;
            }
            else if (results[sub_iter].isUnsafe())
            {
                safe = 1;   // Unsafe
                cout << "Unsafe" << endl;
                break;
            }
            else if (!results[sub_iter].isSafe())
            {
                safe = 2;   // Unknown
                cout << "Unknown" << endl;
                break;
            }
        }
    }


    // Check target
    if (constraints_target != nullptr){
        vector<Constraint> targetSet;
        for (int i = 0; i < constraints_target->size(); i++)
        {
            Constraint c_temp((*constraints_target)[i], vars);
            targetSet.push_back(c_temp);
        }

        bool in_target = 1; 
        for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
        {
            if (!results[sub_iter].fp_end_of_time.isInTarget(targetSet, setting))
            {
                in_target = 0;
                break;
            }
        }
        if (safe == 0 && in_target)
        {
            cout << "VERIFIED" << endl;
        }
        else if (safe == 1)
        {
            cout << "FALSIFIED" << endl;
        }
        else
        {
            cout << "UNKNOWN" << endl;
        }
    }
                                         


    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("time cost: %lf\n", (double)(duration.count() / 1000.0));

    if (test_mode) {
        resultsFile.close();
    }
    return results;
};


