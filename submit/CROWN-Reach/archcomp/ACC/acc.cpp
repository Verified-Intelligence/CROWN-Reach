#include "../../flowstar/flowstar-toolbox/Continuous.h"
#include <jsonrpccpp/client.h>
#include <jsonrpccpp/client/connectors/httpclient.h>
#include <chrono>

using namespace jsonrpc;
using namespace std;
using namespace flowstar;


int main(int argc, char *arsgv[])
{
    // RPC client
    HttpClient httpclient("http://127.0.0.1:5000");
    Client cl(httpclient, JSONRPC_CLIENT_V2);

    // Declaration of variables
    unsigned int numVars = 8;
    unsigned int num_nn_input = 5;
    unsigned int num_nn_output = 1;
    Variables vars;
    int x_lead_id = vars.declareVar("x_lead");
    int v_lead_id = vars.declareVar("v_lead");
    int a_lead_id = vars.declareVar("a_lead");
    int x_ego_id = vars.declareVar("x_ego");
    int v_ego_id = vars.declareVar("v_ego");
    int a_ego_id = vars.declareVar("a_ego");
    int t_id = vars.declareVar("t");
    int a_c_ego_id = vars.declareVar("a_c_ego");

    ODE<Real> dynamics({"v_lead",
                        "a_lead",
                        "-2 * a_lead - 4 - 0.0001 * v_lead^2",
                        "v_ego",
                        "a_ego",
                        "-2 * a_ego + 2 * a_c_ego - 0.0001 * v_ego^2",
                        "1",
                        "0"},
                        vars);
    
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.1, 3);
    setting.setCutoffThreshold(1e-6);
    setting.printOff();
    Interval I(-0.1, 0.1);
    vector<Interval> remainder_estimation(numVars, I);
    setting.setRemainderEstimation(remainder_estimation);

    // Initial set
    int steps = 50;
    Interval init_x_lead(90, 110), init_v_lead(32, 32.2), init_a_lead(0),
             init_x_ego(10, 11), init_v_ego(30, 30.2), init_a_ego(0),
             init_t(0), init_a_c_ego(0);
    vector<Interval> X0;
    X0.push_back(init_x_lead);
    X0.push_back(init_v_lead);
    X0.push_back(init_a_lead);
    X0.push_back(init_x_ego);
    X0.push_back(init_v_ego);
    X0.push_back(init_a_ego);
    X0.push_back(init_t);
    X0.push_back(init_a_c_ego);

    // Translate the initial set to a flowpipe
    Flowpipe initial_set(X0);
    Symbolic_Remainder symbolic_remainder(initial_set, 50);    
    vector<Constraint> safeSet;
    Constraint c_temp("-x_lead + x_ego + 1.4 * v_ego + 10", vars);
    safeSet.push_back(c_temp);
    Result_of_Reachability result;
    int final_result = 0;

    auto start = std::chrono::steady_clock::now();

    for (int iter = 0; iter < steps; iter++)
    {
        cout << "Step " << iter << endl;
        
        Json::Value input_lb(Json::arrayValue);
        Json::Value input_ub(Json::arrayValue);

        // v_set = 30
        input_lb.append(30);
        input_ub.append(30);
        // T_gap = 1.4
        input_lb.append(1.4);
        input_ub.append(1.4);
        // v_ego
        Interval input_range_v_ego;
        initial_set.tmvPre.tms[v_ego_id].intEval(input_range_v_ego, initial_set.domain);
        input_lb.append(input_range_v_ego.inf());
        input_ub.append(input_range_v_ego.sup());
        // D_rel
        TaylorModel<Real> tm_D_rel = initial_set.tmvPre.tms[x_lead_id] - initial_set.tmvPre.tms[x_ego_id];
        Interval input_range_D_rel;
        tm_D_rel.intEval(input_range_D_rel, initial_set.domain);
        input_lb.append(input_range_D_rel.inf());
        input_ub.append(input_range_D_rel.sup());
        // v_rel
        TaylorModel<Real> tm_v_rel = initial_set.tmvPre.tms[v_lead_id] - initial_set.tmvPre.tms[v_ego_id];
        Interval input_range_v_rel;
        tm_v_rel.intEval(input_range_v_rel, initial_set.domain);
        input_lb.append(input_range_v_rel.inf());
        input_ub.append(input_range_v_rel.sup());

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
            // v_set = 30
            tmv_output.tms[j] += 30 * T[j][0];
            // T_gap = 1.4
            tmv_output.tms[j] += 1.4 * T[j][1];
            // v_ego
            tmv_output.tms[j] += initial_set.tmvPre.tms[v_ego_id] * T[j][2];
            // D_rel
            tmv_output.tms[j] += tm_D_rel * T[j][3];
            // v_rel
            tmv_output.tms[j] += tm_v_rel * T[j][4];
            Interval remainder_temp(-interval_r[j], interval_r[j]);
            tmv_output.tms[j].remainder += remainder_temp;
        }

        initial_set.tmvPre.tms[a_c_ego_id] = tmv_output.tms[0];

        // Flow*
        dynamics.reach(result, initial_set, 0.1, setting, safeSet, symbolic_remainder);
        
        if (result.status == COMPLETED_UNSAFE)
        {
            cout << "Unsafe." << endl;
            final_result = 1;
            break;
        }
        else if (result.status == COMPLETED_UNKNOWN)
        {
            cout << "Unknown." << endl;
            final_result = 2;
            break;
        }
        else if (result.status == COMPLETED_SAFE)
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

    // result.transformToTaylorModels(setting);
    // Plot_Setting plot_setting(vars);
    // plot_setting.setOutputDims("t", "x_lead - x_ego - 1.4 * v_ego - 10");
    // plot_setting.plot_2D_octagon_MATLAB("./", "acc_" + to_string(steps), result.tmv_flowpipes, setting);
    
    return 0;
}