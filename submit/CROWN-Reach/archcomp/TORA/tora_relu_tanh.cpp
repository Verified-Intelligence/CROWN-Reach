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
                        "u",
                        "1",
                        "0"},
                        vars);
    
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.01, 6);
    setting.setCutoffThreshold(1e-6);
    setting.printOff();
    Interval I(-0.01, 0.01);
    vector<Interval> remainder_estimation(numVars, I);
    setting.setRemainderEstimation(remainder_estimation);

    // Initial set
    int steps = 10;
    Interval init_x1(-0.77, -0.75), init_x2(-0.45, -0.43), init_x3(0.51, 0.54), init_x4(-0.3, -0.28),
             init_t(0), init_u(0);
    vector<Interval> X0;
    X0.push_back(init_x1);
    X0.push_back(init_x2);
    X0.push_back(init_x3);
    X0.push_back(init_x4);
    X0.push_back(init_t);
    X0.push_back(init_u);

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

        initial_set.tmvPre.tms[u_id] = tmv_output.tms[0];

        // Flow*
        dynamics.reach(result, initial_set, 0.5, setting, safeSet, symbolic_remainder);
        
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
    vector<Constraint> targetSet;
    vector<string> constraints = {"-x1 - 0.1", "x1 - 0.2",
                                  "-x2 - 0.9", "x2 + 0.6"};
    for (int i = 0; i < 4; i++)
    {
        Constraint c_temp(constraints[i], vars);
        targetSet.push_back(c_temp);
    }

    bool b = result.fp_end_of_time.isInTarget(targetSet, setting);

    if (b && final_result == 0)
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

    // result.transformToTaylorModels(setting);
    // Plot_Setting plot_setting(vars);
    // plot_setting.setOutputDims("x1", "x2");
    // plot_setting.plot_2D_octagon_MATLAB("./", "tora_relu_tanh_" + to_string(steps), result.tmv_flowpipes, setting);
    
    return 0;
}