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
    unsigned int numVars = 19;
    unsigned int num_nn_input = 12;
    unsigned int num_nn_output = 6;
    Variables vars;
    int x_id = vars.declareVar("x");
    int y_id = vars.declareVar("y");
    int z_id = vars.declareVar("z");
    int u_id = vars.declareVar("u");
    int v_id = vars.declareVar("v");
    int w_id = vars.declareVar("w");
    int phi_id = vars.declareVar("phi");
    int theta_id = vars.declareVar("theta");
    int psi_id = vars.declareVar("psi");
    int r_id = vars.declareVar("r");
    int p_id = vars.declareVar("p");
    int q_id = vars.declareVar("q");
    int t_id = vars.declareVar("t");
    int Fx_id = vars.declareVar("Fx");
    int Fy_id = vars.declareVar("Fy");
    int Fz_id = vars.declareVar("Fz");
    int Mx_id = vars.declareVar("Mx");
    int My_id = vars.declareVar("My");
    int Mz_id = vars.declareVar("Mz");

    ODE<Real> dynamics({"cos(psi)*cos(theta) * u + (-sin(psi)*cos(phi) + cos(psi)*sin(theta)*sin(phi)) * v + (sin(psi)*sin(phi) + cos(psi)*sin(theta)*cos(phi)) * w",
                        "sin(psi)*cos(theta) * u + (cos(psi)*cos(phi) + sin(psi)*sin(theta)*sin(phi)) * v + (-cos(psi)*sin(phi) + sin(psi)*sin(theta)*cos(phi)) * w",
                        "-sin(theta) * u + cos(theta)*sin(phi) * v + cos(theta)*cos(phi) * w",
                        "-sin(theta) + Fx - q * w + r * v",
                        "cos(theta) * sin(phi) + Fy - r * u + p * w",
                        "cos(theta) * cos(phi) + Fz - p * v + q * u",
                        "(cos(theta) * p + sin(theta)*sin(phi) * q + sin(theta)*cos(phi) * r) / cos(theta)",
                        "(cos(theta)*cos(phi) * q - cos(theta) * sin(phi) * r) / cos(theta)",
                        "(sin(phi) * q + cos(phi) * r) / cos(theta)",
                        "Mz",
                        "Mx",
                        "My",
                        "1", "0", "0", "0", "0", "0", "0"},
                        vars);
    
    Computational_Setting setting(vars);
    setting.setFixedStepsize(0.01, 6);
    setting.setCutoffThreshold(1e-6);
    setting.printOff();
    Interval I(-0.01, 0.01);
    vector<Interval> remainder_estimation(numVars, I);
    setting.setRemainderEstimation(remainder_estimation);

    // Initial set
    int steps = 20;
    Interval init_x(0), init_y(0), init_z(0),
             init_u(1, 1), init_v(1, 1), init_w(1, 1),
             init_phi(0.9, 0.9), init_theta(0.9, 0.9), init_psi(0.9, 0.9),
             init_r(0), init_p(0), init_q(0),
             init_t(0),
             init_Fx(0), init_Fy(0), init_Fz(0),
             init_Mx(0), init_My(0), init_Mz(0);
    vector<Interval> X0;
    X0.push_back(init_x);
    X0.push_back(init_y);
    X0.push_back(init_z);
    X0.push_back(init_u);
    X0.push_back(init_v);
    X0.push_back(init_w);
    X0.push_back(init_phi);
    X0.push_back(init_theta);
    X0.push_back(init_psi);
    X0.push_back(init_r);
    X0.push_back(init_p);
    X0.push_back(init_q);
    X0.push_back(init_t);
    X0.push_back(init_Fx);
    X0.push_back(init_Fy);
    X0.push_back(init_Fz);
    X0.push_back(init_Mx);
    X0.push_back(init_My);
    X0.push_back(init_Mz);

    // Translate the initial set to a flowpipe
    Flowpipe initial_set(X0);
    Symbolic_Remainder symbolic_remainder(initial_set, 1000);

    vector<string> constraints = {"-y - 1", "y - 1",
                                  "-phi - 1", "phi - 1",
                                  "-theta - 1", "theta - 1",
                                  "-psi - 1", "psi - 1"};

    vector<Constraint> safeSet;
    for (int i = 0; i < 8; i++)
    {
        Constraint c_temp(constraints[i], vars);
        safeSet.push_back(c_temp);
    }
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

        initial_set.tmvPre.tms[Fx_id] = tmv_output.tms[0];
        initial_set.tmvPre.tms[Fy_id] = tmv_output.tms[1];
        initial_set.tmvPre.tms[Fz_id] = tmv_output.tms[2];
        initial_set.tmvPre.tms[Mx_id] = tmv_output.tms[3];
        initial_set.tmvPre.tms[My_id] = tmv_output.tms[4];
        initial_set.tmvPre.tms[Mz_id] = tmv_output.tms[5];

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
    // plot_setting.setOutputDims("y", "phi");
    // plot_setting.plot_2D_octagon_MATLAB("./", "airplane_" + to_string(steps), result.tmv_flowpipes, setting);
    
    return 0;
}