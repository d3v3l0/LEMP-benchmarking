/*
 * runAlgoWithTuner.cc
 *
 *  Created on: Feb 10, 2014
 *      Author: chteflio
 */
//    Copyright 2015 Christina Teflioudi
// 
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
// 
//        http://www.apache.org/licenses/LICENSE-2.0
// 
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.


#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <mips/mips.h>

#include <cblas.h>

#define L2_CACHE_SIZE 256000
#define MAX_MEM_SIZE (257840L*1024L*1024L)

using namespace std;
using namespace mips;
using namespace boost::program_options;

inline void computeTopRating(double *ratings_matrix, int *top_K_items,
                             const int num_users, const int num_items) {
  for (int user_id = 0; user_id < num_users; user_id++) {

    unsigned long index = user_id;
    index *= num_items;
    int best_item_id = cblas_idamax(num_items, &ratings_matrix[index], 1);
    top_K_items[user_id] = best_item_id;
  }
}

inline void computeTopK(double *ratings_matrix, int *top_K_items,
                        const int num_users, const int num_items, const int K) {

  for (int i = 0; i < num_users; i++) {

    std::priority_queue<std::pair<double, int>,
                        std::vector<std::pair<double, int> >,
                        std::greater<std::pair<double, int> > > q;

    unsigned long index = i;
    index *= num_items;

    for (int j = 0; j < K; j++) {
      q.push(std::make_pair(ratings_matrix[index + j], j));
    }

    for (int j = K; j < num_items; j++) {
      if (ratings_matrix[index + j] > q.top().first) {
        q.pop();
        q.push(std::make_pair(ratings_matrix[index + j], j));
      }
    }

    for (int j = 0; j < K; j++) {
      const std::pair<double, int> p = q.top();
      top_K_items[i * K + K - 1 - j] = p.second;
      q.pop();
    }
  }
}

inline double decisionRuleBlockedMM(VectorMatrix &q, VectorMatrix &p,
                                    const unsigned int rand_ind,
                                    const unsigned long num_users_per_block,
                                    const int K) {

  double *user_ptr = q.getMatrixRowPtr(rand_ind);
  double *item_ptr = p.getMatrixRowPtr(0);
  const long m = num_users_per_block;
  const int n = p.rowNum;
  const int k = q.colNum;
  const float alpha = 1.0;
  const float beta = 0.0;
  double *matrix_product = (double *)malloc(m * n * sizeof(double));
  int *top_K_items = (int *)malloc(m * K * sizeof(int));

  rg::Timer tt;
  tt.start();
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, user_ptr,
              k, item_ptr, k, beta, matrix_product, n);

  if (K == 1) {
    computeTopRating(matrix_product, top_K_items, m, n);
  } else {
    computeTopK(matrix_product, top_K_items, m, n, K);
  }
  tt.stop();
  free(matrix_product);
  free(top_K_items);
  return (tt.elapsedTime().nanos() / 1E9) / num_users_per_block;
}


int main(int argc, char *argv[]) {
    double theta, R, epsilon, user_sample_ratio;
    string usersFile;
    string itemsFile;
    string logFile, resultsFile;

    bool querySideLeft = true;
    bool isTARR = true;
    int k, cacheSizeinKB, threads, r, m, n;
    std::string methodStr;
    LEMP_Method method;

    // read command line
    options_description desc("Options");
    desc.add_options()
            ("help", "produce help message")
            ("Q^T", value<string>(&usersFile), "file containing the query matrix (left side)")
            ("P", value<string>(&itemsFile), "file containing the probe matrix (right side)")
            ("theta", value<double>(&theta), "theta value")
            ("R", value<double>(&R)->default_value(0.97), "recall parameter for LSH")
            ("x", value<double>(&user_sample_ratio)->default_value(0.0), "user sample ratio")
	    ("epsilon", value<double>(&epsilon)->default_value(0.0), "epsilon value for LEMP-LI with Absolute or Relative Approximation")
            ("querySideLeft", value<bool>(&querySideLeft)->default_value(true), "1 if Q^T contains the queries (default). Interesting for Row-Top-k")
            ("isTARR", value<bool>(&isTARR)->default_value(true), "for LEMP-TA. If 1 Round Robin schedule is used (default). Otherwise Max PiQi")
            ("method", value<string>(&methodStr), "LEMP_X where X: L, LI, LC, I, C, TA, TREE, AP, LSH")
            ("k", value<int>(&k)->default_value(0), "top k (default 0). If 0 Above-theta will run")
            ("logFile", value<string>(&logFile)->default_value(""), "output File (contains runtime information)")
	    ("resultsFile", value<string>(&resultsFile)->default_value(""), "output File (contains the results)")
            ("cacheSizeinKB", value<int>(&cacheSizeinKB)->default_value(8192), "cache size in KB")
            ("t", value<int>(&threads)->default_value(1), "num of threads (default 1)")
            ("r", value<int>(&r)->default_value(0), "num of coordinates in each vector (needed when reading from csv files)")
            ("m", value<int>(&m)->default_value(0), "num of vectors in Q^T (needed when reading from csv files)")
            ("n", value<int>(&n)->default_value(0), "num of vectors in P (needed when reading from csv files)")
            ;

    positional_options_description pdesc;
    pdesc.add("Q^T", 1);
    pdesc.add("P", 2);

    variables_map vm;
    store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
    notify(vm);

    if (vm.count("help") || vm.count("Q^T") == 0 || vm.count("P") == 0) {
        cout << "runLemp [options] <Q^T> <P>" << endl << endl;
        cout << desc << endl;
        return 1;
    }
        
    InputArguments args;
    args.logFile = logFile;
    args.theta = theta;
    args.k = k;
    args.threads = threads;

    if (methodStr.compare("LEMP_LI") == 0) {
        method = LEMP_LI;
    } else if (methodStr.compare("LEMP_LC") == 0) {
        method = LEMP_LC;
    } else if (methodStr.compare("LEMP_L") == 0) {
        method = LEMP_L;
    } else if (methodStr.compare("LEMP_I") == 0) {
        method = LEMP_I;
    } else if (methodStr.compare("LEMP_C") == 0) {
        method = LEMP_C;
    } else if (methodStr.compare("LEMP_TA") == 0) {
        method = LEMP_TA;
    } else if (methodStr.compare("LEMP_TREE") == 0) {
        method = LEMP_TREE;
    } else if (methodStr.compare("LEMP_AP") == 0) {
        method = LEMP_AP;
    } else if (methodStr.compare("LEMP_LSH") == 0) {
        method = LEMP_LSH;
    } else if (methodStr.compare("LEMP_BLSH") == 0) {
        method = LEMP_BLSH;
    } 
    else {
        cout << "[ERROR] This method is not possible. Please try {LEMP_L, LEMP_LI, LEMP_LC, LEMP_I, LEMP_C, LEMP_TA, LEMP_TREE, LEMP_AP, LEMP_LSH, LEMP_BLSH}" << endl << endl;
        cout << desc << endl;
        return 1;
    }

    VectorMatrix leftMatrix, rightMatrix;

    if (querySideLeft) {
        leftMatrix.readFromFile(usersFile, r, m, true);
        rightMatrix.readFromFile(itemsFile, r, n, false);
    } else {
        leftMatrix.readFromFile(itemsFile, r, n, false);
        rightMatrix.readFromFile(usersFile, r, m, true);
    }

    mips::Lemp algo(args, cacheSizeinKB, method, isTARR, R, epsilon);
    
    algo.initialize(rightMatrix);

    Results results;
    if (args.k > 0) {
#ifdef ONLINE_DECISION_RULE
    std::random_device rd; // only used once to initialise (seed) engine
    std::mt19937 rng(
        rd()); // random-number engine used (Mersenne-Twister in this case)
    unsigned long num_users_per_block = 0;
    if (user_sample_ratio == 0.0) {
      // Default
      num_users_per_block =
          4 * L2_CACHE_SIZE / (sizeof(double) * leftMatrix.colNum);
      while (num_users_per_block * rightMatrix.rowNum * sizeof(double) > MAX_MEM_SIZE) {
        num_users_per_block /= 2;
      }
    } else {
      num_users_per_block = (long)(user_sample_ratio * leftMatrix.rowNum);
    }
    std::uniform_int_distribution<int> uni(
        0, leftMatrix.rowNum - num_users_per_block); // guaranteed unbiased
    const unsigned int rand_ind = uni(rng);

    const double blocked_mm_time =
        decisionRuleBlockedMM(leftMatrix, rightMatrix, rand_ind, num_users_per_block, args.k);

    double *sample_ptr = leftMatrix.getMatrixRowPtr(rand_ind);
    double *new_ptr = (double *)malloc(num_users_per_block * leftMatrix.colNum * sizeof(double));
    std::memcpy(new_ptr, sample_ptr, num_users_per_block * leftMatrix.colNum * sizeof(double));


    VectorMatrix sampleLeftMatrix(new_ptr, leftMatrix.colNum, num_users_per_block);
    
    rg::Timer tt;
    tt.start();
    // sample using rand_ind and num_users_per_block
    algo.runTopK(sampleLeftMatrix, results);
    tt.stop();

    const double lemp_time = (tt.elapsedTime().nanos() / 1E9) / num_users_per_block;

    algo.addSampleStats(user_sample_ratio, blocked_mm_time, lemp_time);
    cout << "Blocked MM time: " << blocked_mm_time << "s" << endl;
    cout << "LEMP time: " << lemp_time << "s" << endl;
    if (blocked_mm_time < lemp_time) {
      cout << "Blocked MM wins" << endl;
    } else {
      cout << "LEMP wins" << endl;
#ifndef TEST_ONLY
      // TODO: run it on everything else [0-rand_ind), [rand_ind +
      // num_users_per_block, num_users) and output results

      algo.runTopK(leftMatrix, results);
#endif
    }
    algo.outputStats();
#else
      algo.runTopK(leftMatrix, results);
      algo.outputStats();
#endif
    } else {
        algo.runAboveTheta(leftMatrix, results);
        algo.outputStats();
    }
    
    if (resultsFile != "") {
        results.writeToFile(resultsFile);
    }

    return 0;
}


