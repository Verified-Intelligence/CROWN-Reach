#include "../../flowstar/flowstar-toolbox/Continuous.h"
#include <jsonrpccpp/client.h>
#include <jsonrpccpp/client/connectors/httpclient.h>
#include <chrono>
using namespace jsonrpc;
using namespace std;
using namespace flowstar;


int main(int argc, char *argv[])
{
    // RPC client
    HttpClient httpclient("http://127.0.0.1:5000");
    Client cl(httpclient, JSONRPC_CLIENT_V2);

    // Declaration of variables
    unsigned int numVars = 10;
    unsigned int num_nn_input = 6;
    unsigned int num_nn_output = 3;
    Variables vars;
    int x1_id = vars.declareVar("x1");
    int x2_id = vars.declareVar("x2");
    int x3_id = vars.declareVar("x3");
    int x4_id = vars.declareVar("x4");
    int x5_id = vars.declareVar("x5");
    int x6_id = vars.declareVar("x6");
    int t_id = vars.declareVar("t");
    int u1_id = vars.declareVar("u1");
    int u2_id = vars.declareVar("u2");
    int u3_id = vars.declareVar("u3");


    ODE<Real> dynamics({"0.25 * (u1 + x2 * x3)",
                        "0.5 * (u2 - 3 * x1 * x3)",                        
                        "u3 + 2 * x1 * x2",
                        "0.5 * (x2 * (x4^2 + x5^2 + x6^2 - x6) + x3 * (x4^2 + x5^2 + x5 + x6^2) + x1 * (x4^2 + x5^2 + x6^2 + 1))",
                        "0.5 * (x1 * (x4^2 + x5^2 + x6^2 + x6) + x3 * (x4^2 - x4 + x5^2 + x6^2) + x2 * (x4^2 + x5^2 + x6^2 + 1))",
                        "0.5 * (x1 * (x4^2 + x5^2 - x5 + x6^2) + x2 * (x4^2 + x4 + x5^2 + x6^2) + x3 * (x4^2 + x5^2 + x6^2 + 1))",
                        "1", "0", "0", "0"},
                        vars);
    
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.05, 3);
    setting.setCutoffThreshold(1e-6);
    setting.printOff();
    Interval I(-0.01, 0.01);
    vector<Interval> remainder_estimation(numVars, I);
    setting.setRemainderEstimation(remainder_estimation);

    // Initial set
    int steps = 30;
    Interval init_x1(-0.45, -0.44), init_x2(-0.55, -0.54), init_x3(0.65, 0.66),
             init_x4(-0.75, -0.74), init_x5(0.85, 0.86), init_x6(-0.65, -0.64),
             init_t(0), init_u1(0), init_u2(0), init_u3(0);
    vector<Interval> X0;
    X0.push_back(init_x1);
    X0.push_back(init_x2);
    X0.push_back(init_x3);
    X0.push_back(init_x4);
    X0.push_back(init_x5);
    X0.push_back(init_x6);
    X0.push_back(init_t);
    X0.push_back(init_u1);
    X0.push_back(init_u2);
    X0.push_back(init_u3);

    // Translate the initial set to a flowpipe
    Flowpipe initial_set(X0);
    Symbolic_Remainder symbolic_remainder(initial_set, 1000);

    vector<Constraint> safeSet;
    Result_of_Reachability result;
    int final_result = 0;

    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < steps; iter++)
    {
        cout << "Step " << iter << endl;
        
        Json::Value input_lb(Json::arrayValue);
        Json::Value input_ub(Json::arrayValue);
        for (int i = 0; i < num_nn_input; i++)
        {
            Interval input_range_temp;
            initial_set.tmvPre.tms[i].intEval(input_range_temp, initial_set.domain);
            input_lb.append(input_range_temp.inf());
            input_ub.append(input_range_temp.sup());
        }

        // Call CROWN
        Json::Value params, output_coefficients;
        params["input_lb"] = input_lb;
        params["input_ub"] = input_ub;
        cl.CallMethod("CROWN_reach", params, output_coefficients);
        // Unpack results from CROWN
        Matrix<Real> T(num_nn_output, num_nn_input, Real(0));
        vector<Real> c_vector;
        vector<double> interval_r;
        for (int j = 0; j < num_nn_output; j++)
        {
            for (int i = 0; i < num_nn_input; i++)
            {
                T[j][i] = output_coefficients["T"][j][i].asFloat();
            }
            double u_max = output_coefficients["u_max"][j].asFloat();
            double u_min = output_coefficients["u_min"][j].asFloat();
            c_vector.push_back((u_max + u_min) / 2);
            interval_r.push_back((u_max - u_min) / 2);
        }
        
        // Construct new Taylor Models
        TaylorModelVec<Real> tmv_output(c_vector, numVars);
        for (int j = 0; j < num_nn_output; j++)
        {
            for (int i = 0; i < num_nn_input; i++)
            {
                tmv_output.tms[j] += initial_set.tmvPre.tms[i] * T[j][i];
            }
            Interval remainder_temp(-interval_r[j], interval_r[j]);
            tmv_output.tms[j].remainder += remainder_temp;
        }

        initial_set.tmvPre.tms[u1_id] = tmv_output.tms[0];
        initial_set.tmvPre.tms[u2_id] = tmv_output.tms[1];
        initial_set.tmvPre.tms[u3_id] = tmv_output.tms[2];

        // Flow*
        dynamics.reach(result, initial_set, 0.1, setting, safeSet, symbolic_remainder);
        if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
        {
            initial_set = result.fp_end_of_time;
        }
        else
        {
            cout << "Flow* terminated." << endl;
            final_result = 2;
            break;
        }
    }

    // Check reachability
    vector<Constraint> unsafeSet;
    vector<string> constraints = {"-x1 - 0.2", "x1",
                                  "-x2 - 0.5", "x2 + 0.4",
                                  "-x3", "x3 - 0.2",
                                  "-x4 - 0.4", "x4 + 0.6",
                                  "-x5 + 0.7", "x5 - 0.8",
                                  "-x6 - 0.4", "x6 + 0.2"};
    for (int i = 0; i < 12; i++)
    {
        Constraint c_temp(constraints[i], vars);
        unsafeSet.push_back(c_temp);
    }
    result.unsafetyChecking(unsafeSet, setting.tm_setting, setting.g_setting);
    if (result.isSafe() && final_result != 2)
    {
        cout << "VERIFIED" << endl;
    }
    else if (result.isUnsafe() && final_result != 2)
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

    // result.transformToTaylorModels(setting);
    // Plot_Setting plot_setting(vars);
    // plot_setting.setOutputDims("x1", "x2");
    // plot_setting.plot_2D_octagon_MATLAB("./", "attitude_control_" + to_string(steps), result.tmv_flowpipes, setting);
    
    return 0;
}