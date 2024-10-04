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
    unsigned int numVars = 16;
    unsigned int num_nn_input = 12;
    unsigned int num_nn_output = 3;
    Variables vars;
    int x_ids[num_nn_input];
    for (int i = 1; i <= num_nn_input; i++)
    {
        string var_name = "x" + to_string(i);
        int var_id = vars.declareVar(var_name);
        x_ids[i-1] = var_id;
    }
    int t_id = vars.declareVar("t");
    int u_ids[num_nn_output];
    for (int i = 1; i <= num_nn_output; i++)
    {
        string var_name = "u" + to_string(i);
        int var_id = vars.declareVar(var_name);
        u_ids[i-1] = var_id;
    }


    ODE<Real> dynamics({"cos(x8)*cos(x9)*x4 + (sin(x7)*sin(x8)*cos(x9) - cos(x7)*sin(x9))*x5 + (cos(x7)*sin(x8)*cos(x9) + sin(x7)*sin(x9))*x6",
                        "cos(x8)*sin(x9)*x4 + (sin(x7)*sin(x8)*sin(x9) - cos(x7)*cos(x9))*x5 + (cos(x7)*sin(x8)*sin(x9) + sin(x7)*cos(x9))*x6",
                        "sin(x8)*x4 - sin(x7)*cos(x8)*x5 - cos(x7)*cos(x8)*x6",
                        "x12*x5 * x11*x6 - 9.81 *sin(x8)",
                        "x10*x6 - x11*x6 - 9.81 *sin(x8)",
                        "x11*x4 - x10*x5 + 9.81 *cos(x8)*cos(x7) - 9.81 - u1 / 1.4",
                        "x10 + sin(x7)*sin(x8)/cos(x8)*x11 + cos(x7)*sin(x8)/cos(x8)*x12",
                        "cos(x7)*x11 - sin(x7)*x12",
                        "sin(x7)*x11/cos(x8) - cos(x7)*x12/cos(x8)",
                        "x11*x12*(0.054 - 0.104) / 0.054 + u2 / 0.054",
                        "(0.104 - 0.054)*x10*x12 / 0.054 + u3 / 0.054",
                        "0",
                        "1", "0", "0", "0"},
                        vars);
    
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.005, 2);
    setting.setCutoffThreshold(1e-6);
    setting.printOff();
    Interval I(-0.1, 0.1);
    vector<Interval> remainder_estimation(numVars, I);
    setting.setRemainderEstimation(remainder_estimation);
    vector<Constraint> safeSet;

    // Initial set
    int steps = 50;
    vector<Interval> X0;
    Interval init_x1(-0.4, 0.4), init_x2(-0.4, 0.4), init_x3(-0.4, 0.4),
             init_x4(-0.4, 0.4), init_x5(-0.4, 0.4), init_x6(-0.4, 0.4),
             init_x7(0), init_x8(0), init_x9(0), init_x10(0), init_x11(0), init_x12(0),
             init_t(0), init_u1(0), init_u2(0), init_u3(0);

    list<Interval> list_x1;
    init_x1.split(list_x1, 8);
    list<Interval> list_x2;
    init_x2.split(list_x2, 8);
    list<Interval> list_x3;
    init_x3.split(list_x3, 8);
    list<Interval> list_x4;
    init_x4.split(list_x4, 2);
    list<Interval> list_x5;
    init_x5.split(list_x5, 1);
    list<Interval> list_x6;
    init_x6.split(list_x6, 1);

    vector<Flowpipe> initial_sets;
    vector<vector<Interval> > initial_boxes;
    for (auto iter1 = list_x1.begin(); iter1 != list_x1.end(); ++iter1)
    {
        for (auto iter2 = list_x2.begin(); iter2 != list_x2.end(); ++iter2)
        {
            for (auto iter3 = list_x3.begin(); iter3 != list_x3.end(); ++iter3)
            {
                for (auto iter4 = list_x4.begin(); iter4 != list_x4.end(); ++iter4)
                {
                    for (auto iter5 = list_x5.begin(); iter5 != list_x5.end(); ++iter5)
                    {
                        for (auto iter6 = list_x6.begin(); iter6 != list_x6.end(); ++iter6)
                        {
                            vector<Interval> X0;
                            X0.push_back(*iter1);
                            X0.push_back(*iter2);
                            X0.push_back(*iter3);
                            X0.push_back(*iter4);
                            X0.push_back(*iter5);
                            X0.push_back(*iter6);
                            X0.push_back(init_x7);
                            X0.push_back(init_x8);
                            X0.push_back(init_x9);
                            X0.push_back(init_x10);
                            X0.push_back(init_x11);
                            X0.push_back(init_x12);
                            X0.push_back(init_t);
                            X0.push_back(init_u1);
                            X0.push_back(init_u2);
                            X0.push_back(init_u3);
                            Flowpipe initial_set_temp(X0);
                            initial_sets.push_back(initial_set_temp);
                            initial_boxes.push_back(X0);
                        }
                    }
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

    auto start = std::chrono::steady_clock::now();
    unsigned int num_threads = boost::thread::hardware_concurrency();

    for (int iter = 0; iter < steps; iter++)
    {
        boost::asio::thread_pool pool(num_threads);
        cout << "Step " << iter << endl;
        cout << "Constructing input." << endl;
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
        cout << "Calling CROWN." << endl;
        cl.CallMethod("CROWN_reach", params, output_coefficients);
        // Unpack results from CROWN
        cout << "Unpacking output from CROWN." << endl;
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
        cout << "Flow* started." << endl;
        std::atomic<bool> terminate(false);
        for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
        {
            boost::asio::post(pool, [&, sub_iter] {
                dynamics.reach(results[sub_iter], initial_sets[sub_iter], 0.1, setting, safeSet, symbolic_remainders[sub_iter]);
                if (results[sub_iter].status == COMPLETED_SAFE || results[sub_iter].status == COMPLETED_UNSAFE || results[sub_iter].status == COMPLETED_UNKNOWN)
                {
                    initial_sets[sub_iter] = results[sub_iter].fp_end_of_time;
                }
                else
                {
                    cout << "Flow* terminated." << endl;
                    cout << "Broken branch: " << sub_iter << endl;
                    cout << initial_boxes[sub_iter][0] << "\t" << initial_boxes[sub_iter][1] << "\t" << initial_boxes[sub_iter][2] << "\t" 
                         << initial_boxes[sub_iter][3] << "\t" << initial_boxes[sub_iter][4] << "\t" << initial_boxes[sub_iter][5] << endl;
                    terminate.store(true);
                }
            });
        }
        pool.join(); 
        if (terminate.load())
        {
            break;
        }
        cout << "Flow* finished." << endl;
        // auto end_temp = std::chrono::steady_clock::now();
        // auto duration_temp = std::chrono::duration_cast<std::chrono::milliseconds>(end_temp - start);
        // printf("time cost so far: %lf\n", (double)(duration_temp.count() / 1000.0));
    }


    vector<Constraint> targetSet;
    vector<string> constraints = {"-x3 + 0.94", "x3 - 1.06"};

    for (int i = 0; i < 2; i++)
    {
        Constraint c_temp(constraints[i], vars);
        targetSet.push_back(c_temp);
    }

    bool is_reachable = 1;
    for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
    {
        bool b = results[sub_iter].fp_end_of_time.isInTarget(targetSet, setting);
        if (!b)
        {
            is_reachable = 0;
            cout << "Unknown branch:" << endl;
            cout << initial_boxes[sub_iter][0] << "\t" << initial_boxes[sub_iter][1] << "\t" << initial_boxes[sub_iter][2] << "\t" 
                 << initial_boxes[sub_iter][3] << "\t" << initial_boxes[sub_iter][4] << "\t" << initial_boxes[sub_iter][5] << endl;
            break;
        }
    }

    if (is_reachable)
    {
        cout << "VERIFIED" << endl;
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
    // plot_setting.setOutputDims("t", "x3");
    // plot_setting.plot_2D_octagon_MATLAB("./", "quad_" + to_string(steps), combined_results, setting);
    
    return 0;
}