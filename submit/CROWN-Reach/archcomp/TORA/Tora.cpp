#include "../../flowstar/flowstar-toolbox/Continuous.h"
#include <jsonrpccpp/client.h>
#include <jsonrpccpp/client/connectors/httpclient.h>
#include <boost/thread/thread.hpp>
#include <chrono>

// Temporarily undefine PN to avoid conflict with Boost Asio
#ifdef PN
#undef PN
#endif
#include <boost/asio.hpp>

// Redefine PN after Boost Asio headers are included
#ifdef PN_CONFLICT
#define PN PN_CONFLICT
#else
#define PN 1
#endif

using namespace jsonrpc;
using namespace std;
using namespace flowstar;


int main(int argc, char *argv[])
{
    // RPC client
    HttpClient httpclient("http://127.0.0.1:5000");
    Client cl(httpclient, JSONRPC_CLIENT_V2);

    // Declaration of variables
    unsigned int numVars = 6;
    unsigned int num_nn_input = 4;
    unsigned int num_nn_output = 1;
    Variables vars;
    int x1_id = vars.declareVar("x1");
    int x2_id = vars.declareVar("x2");
    int x3_id = vars.declareVar("x3");
    int x4_id = vars.declareVar("x4");
    int t_id = vars.declareVar("t");
    int u_id = vars.declareVar("u");

    ODE<Real> dynamics({"x2",
                        "-x1 + 0.1 * sin(x3)",
                        "x4",
                        "u - 10",
                        "1",
                        "0"},
                        vars);
    
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.1, 3);
    setting.setCutoffThreshold(1e-6);
    setting.printOff();
    Interval I(-0.01, 0.01);
    vector<Interval> remainder_estimation(numVars, I);
    setting.setRemainderEstimation(remainder_estimation);

    // Initial set
    int steps = 20;
    Interval init_x1(0.6, 0.7), init_x2(-0.7, -0.6), init_x3(-0.4, -0.3), init_x4(0.5, 0.6),
             init_t(0), init_u(0);
    vector<string> constraints = {"-x1 - 2", "x1 - 2",
                                  "-x2 - 2", "x2 - 2",
                                  "-x3 - 2", "x3 - 2",
                                  "-x4 - 2", "x4 - 2"};
    vector<Constraint> safeSet;
    for (int i = 0; i < 8; i++)
    {
        Constraint c_temp(constraints[i], vars);
        safeSet.push_back(c_temp);
    }

    list<Interval> list_x1;
    init_x1.split(list_x1, 4);
    list<Interval> list_x2;
    init_x2.split(list_x2, 3);
    list<Interval> list_x3;
    init_x3.split(list_x3, 1);
    list<Interval> list_x4;
    init_x4.split(list_x4, 1);

    vector<Flowpipe> initial_sets;
    for (auto iter1 = list_x1.begin(); iter1 != list_x1.end(); ++iter1)
    {
        for (auto iter2 = list_x2.begin(); iter2 != list_x2.end(); ++iter2)
        {
            for (auto iter3 = list_x3.begin(); iter3 != list_x3.end(); ++iter3)
            {
                for (auto iter4 = list_x4.begin(); iter4 != list_x4.end(); ++iter4)
                {
                    vector<Interval> X0;
                    X0.push_back(*iter1);
                    X0.push_back(*iter2);
                    X0.push_back(*iter3);
                    X0.push_back(*iter4);
                    X0.push_back(init_t);
                    X0.push_back(init_u);
                    Flowpipe initial_set_temp(X0);
                    initial_sets.push_back(initial_set_temp);
                }
            }
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
    int final_result = 0;
    auto start = std::chrono::steady_clock::now();
    unsigned int num_threads = boost::thread::hardware_concurrency();

    for (int iter = 0; iter < steps; iter++)
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

            initial_sets[sub_iter].tmvPre.tms[u_id] = tmv_output.tms[0];
        }

        // Flow*
        std::atomic<bool> terminate(false);
        for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
        {
            boost::asio::post(pool, [&, sub_iter]{
            dynamics.reach(results[sub_iter], initial_sets[sub_iter], 1, setting, safeSet, symbolic_remainders[sub_iter]);
            if (results[sub_iter].status == COMPLETED_UNSAFE)
            {
                cout << "Unsafe." << endl;
                final_result = 1;
                terminate.store(true);
            }
            else if (results[sub_iter].status == COMPLETED_UNKNOWN)
            {
                cout << "Unknown." << endl;
                final_result = 2;
                terminate.store(true);
            }
            else if (results[sub_iter].status == COMPLETED_SAFE)
            {
                initial_sets[sub_iter] = results[sub_iter].fp_end_of_time;
            }
            else
            {
                cout << "Flow* terminated." << endl;
                final_result = 2;
                terminate.store(true);
            }    
            });
        }
        pool.join(); 
        if (terminate.load())
        {
            break;
        }
    }
    if (final_result == 0)
    {
        cout << "VERIFIED" << endl;
    }
    else if (final_result == 1)
    {
        cout << "FALSIFIED" << endl;
    }
    else
    {
        cout << "UNKNOWN" << endl;
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("time cost: %lf\n", (double)(duration.count() / 1000.0));

    // TaylorModelFlowpipes combined_results;
    // for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
    // {
    //     results[sub_iter].transformToTaylorModels(setting);
    //     combined_results.tmv_flowpipes.insert(
    //         combined_results.tmv_flowpipes.end(), 
    //         results[sub_iter].tmv_flowpipes.tmv_flowpipes.begin(), 
    //         results[sub_iter].tmv_flowpipes.tmv_flowpipes.end());
    // }
    // Plot_Setting plot_setting(vars);
    // plot_setting.setOutputDims("x1", "x2");
    // plot_setting.plot_2D_octagon_MATLAB("./", "Tora_x1_x2_" + to_string(steps), combined_results, setting);
    // plot_setting.setOutputDims("x3", "x4");
    // plot_setting.plot_2D_octagon_MATLAB("./", "Tora_x3_x4_" + to_string(steps), combined_results, setting);
    return 0;
}