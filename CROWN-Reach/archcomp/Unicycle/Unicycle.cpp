/*
 * Copyright (c) 2024, The CROWN-Reach Team
 * Primary contacts:
 *   Xiangru Zhong <xiangruzh0915@gmail.com>
 *   Yuhao Jia <yuhaojia98@g.ucla.edu>
 *   Huan Zhang <huan@huan-zhang.com>
 *
 * This file is part of the CROWN-Reach verifier.
 *
 * This program is licensed under the BSD 3-Clause License,
 * contained in the LICENCE file in this directory.
 */

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
    unsigned int numVars = 8;
    unsigned int num_nn_input = 4;
    unsigned int num_nn_output = 2;
    Variables vars;
    int x1_id = vars.declareVar("x1");
    int x2_id = vars.declareVar("x2");
    int x3_id = vars.declareVar("x3");
    int x4_id = vars.declareVar("x4");
    int t_id = vars.declareVar("t");
    int u1_id = vars.declareVar("u1");
    int u2_id = vars.declareVar("u2");
    int w_id = vars.declareVar("w");

    ODE<Real> dynamics({"x4 * cos(x3)",
                        "x4 * sin(x3)",
                        "u2 + w - 20",
                        "u1 + w - 20",
                        "1",
                        "0",
                        "0",
                        "0"},
                        vars);
    
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.02, 2);
    setting.setCutoffThreshold(1e-6);
    setting.printOff();
    Interval I(-0.01, 0.01);
    vector<Interval> remainder_estimation(numVars, I);
    setting.setRemainderEstimation(remainder_estimation);

    // Initial set
    int steps = 50;
    Interval init_x1(9.5, 9.55), init_x2(-4.5, -4.45), init_x3(2.1, 2.11), init_x4(1.5, 1.51),
             init_t(0), init_u1(0), init_u2(0), init_w(-1e-4, 1e-4);
    vector<Interval> X0;
    X0.push_back(init_x1);
    X0.push_back(init_x2);
    X0.push_back(init_x3);
    X0.push_back(init_x4);
    X0.push_back(init_t);
    X0.push_back(init_u1);
    X0.push_back(init_u2);
    X0.push_back(init_w);

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

        // Flow*
        dynamics.reach(result, initial_set, 0.2, setting, safeSet, symbolic_remainder);
        
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
    vector<string> constraints = {"x1 - 0.6", "-x1 - 0.6",
                                  "x2 - 0.2", "-x2 - 0.2",
                                  "x3 - 0.06", "-x3 - 0.06",
                                  "x4 - 0.3", "-x4 - 0.3"};
    for (int i = 0; i < 8; i++)
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
    // plot_setting.plot_2D_octagon_MATLAB("./", "Unicycle_" + to_string(steps), result.tmv_flowpipes, setting);
    
    return 0;
}