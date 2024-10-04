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

#include "CrownReach.h"
#include <fstream>
#include <json/json.h>
#include <map>
#include <yaml-cpp/yaml.h>


int main(int argc, char *argv[])
{
    //---------------- read config file ----------------//
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config file path>" << std::endl;
        return 1;
    }
    string configFilePath = argv[1];
    YAML::Node config = YAML::LoadFile(configFilePath);

    //---------------- check test mode ----------------//
    bool test_mode = false;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--test") {
            test_mode = true;
            std::cout << "Running in test mode: only first 5 epochs will be executed." << std::endl;
        }
    }

    //---------------- System Setting ----------------//
    unsigned int numVars = config["num_vars"].as<unsigned int>();
    unsigned int num_nn_input = config["num_nn_input"].as<unsigned int>();
    unsigned int num_nn_output = config["num_nn_output"].as<unsigned int>();
    vector<unsigned int> sys_settings = {numVars,num_nn_input,num_nn_output};

    unsigned int steps = config["steps"].as<unsigned int>();
    double step_size = config["step_size"].as<double>();

    // ---------------- Declaration of variables ----------------//
    Variables vars;
    for (int i = 1; i <= numVars-num_nn_output-1; i++)
    {
        string var_name = config["initial_set"][i-1]["name"].as<string>();
        int var_id = vars.declareVar(var_name);
    }
    int t_id = vars.declareVar("t");
    int u_ids[num_nn_output];
    for (int i = 1; i <= num_nn_output; i++)
    {
        string var_name = config["initial_set"][numVars-num_nn_output-1+i]["name"].as<string>();
        int var_id = vars.declareVar(var_name);
        u_ids[i-1] = var_id;
    }
    // int other_ids[numVars-num_nn_input-num_nn_output-1];
    // for (int i = 1; i <= numVars-num_nn_input-num_nn_output-1; i++)
    // {
    //     string var_name = config["initial_set"][num_nn_input+num_nn_output+i]["name"].asString();
    //     int var_id = vars.declareVar(var_name);
    //     other_ids[i-1] = var_id;
    // }

    //----------------  Define ODE ----------------//
    vector<string> dynamics_expressions;
    for (const auto& expr : config["dynamics_expressions"])
    {
        dynamics_expressions.push_back(expr.as<string>());
    }

    ODE<Real> dynamics(dynamics_expressions, vars);
    //ODE settings
    Computational_Setting setting(vars);
    setting.setFixedStepsize(config["ode_step_size"].as<double>(), config["ode_order"].as<unsigned int>());
    setting.setCutoffThreshold(config["cut_off_threshold"].as<double>());
    setting.printOff();
    Interval I(config["remainder_estimation"][0].as<double>(), config["remainder_estimation"][1].as<double>());
    vector<Interval> remainder_estimation(numVars, I);
    setting.setRemainderEstimation(remainder_estimation);

    //---------------- Initial set ----------------//
    vector<pair<string, Interval>> init_intervals;
    const YAML::Node& initial_set = config["initial_set"];
    for (int i = 0; i < initial_set.size(); ++i) {
        string var_name = initial_set[i]["name"].as<string>();
        double lower = initial_set[i]["interval"][0].as<double>();
        double upper = initial_set[i]["interval"][1].as<double>();
        Interval interval(lower, upper);
        init_intervals.push_back(make_pair(var_name, interval));
    }


    // Split initial set (optional)
    vector<Flowpipe> initial_sets;
    vector<vector<Interval> > initial_boxes;
    map<string, list<Interval>> split_lists;

    if (config["split_vars"]) {
        vector<string> split_vars;
        for (const auto& var : config["split_vars"]) {
            split_vars.push_back(var.as<string>());
        }
        
        for (const auto& var : split_vars) {
            auto it = std::find_if(init_intervals.begin(), init_intervals.end(), [&](const pair<string, Interval>& p){
                return p.first == var;
            });

            if (it != init_intervals.end()) {
                int idx = static_cast<int>(std::distance(init_intervals.begin(), it));
                int split_count = initial_set[idx]["splits"].as<int>();
                it->second.split(split_lists[var], split_count);
            }
        }

        function<void(int, vector<Interval>&)> split_recursive = [&](int idx, vector<Interval>& current_intervals) {
            if (idx == split_vars.size()) {
                vector<Interval> X0 = current_intervals;
                for (const auto& var : init_intervals) {
                    if (std::find(split_vars.begin(), split_vars.end(), var.first) == split_vars.end()) {
                        X0.push_back(var.second);
                    }
                }
                Flowpipe initial_set_temp(X0);
                initial_sets.push_back(initial_set_temp);
                initial_boxes.push_back(X0);
                return;
            }

            auto it = split_lists.find(split_vars[idx]);
            if (it != split_lists.end()) {
                for (const auto& interval : it->second) {
                    vector<Interval> new_intervals = current_intervals;
                    new_intervals.push_back(interval);
                    split_recursive(idx + 1, new_intervals);
                }
            }
        };
        vector<Interval> empty_intervals;
        split_recursive(0, empty_intervals);
    }
    else{
        vector<Interval> X0;
        for (const auto& var : init_intervals)
        {
            // cout << "Variable: " << var.first << " - Interval: [" << var.second.inf() << ", " << var.second.sup() << "]" << endl;
            X0.push_back(var.second);
        }
        Flowpipe initial_set_temp(X0);
        initial_sets.push_back(initial_set_temp);
    }


    //---------------- Task Conditions ----------------//
    vector<string> constraints_unsafe;
    if (config["constraints_unsafe"]) {
        for (const auto& constraint : config["constraints_unsafe"]) {
            constraints_unsafe.push_back(constraint.as<string>());
        }
    }
    vector<string> constraints_safe;
    if (config["constraints_safe"]) {
        for (const auto& constraint : config["constraints_safe"]) {
            constraints_safe.push_back(constraint.as<string>());
        }
    }
    vector<string> constraints_target;
    if (config["constraints_target"]) {
        for (const auto& constraint : config["constraints_target"]) {
            constraints_target.push_back(constraint.as<string>());
        }
    }



    vector<Result_of_Reachability> results = CrownReach(initial_sets, 
                                                        dynamics, 
                                                        setting, 
                                                        vars,
                                                        sys_settings, 
                                                        steps,
                                                        step_size,
                                                        u_ids, 
                                                        initial_boxes.empty() ? nullptr : &initial_boxes,
                                                        constraints_safe.empty() ? nullptr : &constraints_safe,
                                                        constraints_unsafe.empty() ? nullptr : &constraints_unsafe,
                                                        constraints_target.empty() ? nullptr : &constraints_target,
                                                        test_mode);


    //---------------- Plotting ----------------//                     
    TaylorModelFlowpipes combined_results;
    for (int sub_iter = 0; sub_iter < initial_sets.size(); sub_iter++)
    {
        results[sub_iter].transformToTaylorModels(setting);
        combined_results.tmv_flowpipes.insert(
            combined_results.tmv_flowpipes.end(), 
            results[sub_iter].tmv_flowpipes.tmv_flowpipes.begin(), 
            results[sub_iter].tmv_flowpipes.tmv_flowpipes.end());
    }
    if (config["plot_vars"] && config["plot_name"])
    {
        Plot_Setting plot_setting(vars);
        plot_setting.setOutputDims(config["plot_vars"][0].as<string>(), config["plot_vars"][1].as<string>());
        plot_setting.plot_2D_octagon_MATLAB("./", config["plot_name"].as<string>() + "_" + std::to_string(steps), combined_results, setting);
    }
}