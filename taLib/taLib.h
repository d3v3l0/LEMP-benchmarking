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

#ifndef TA_TALIB_H
#define TA_TALIB_H


#include <omp.h>

#include <util/random.h>

#include <taLib/structs/Definitions.h>
#include <taLib/structs/read.h>
#include <taLib/structs/BasicStructs.h>
#include <taLib/structs/Args.h>

#include <taLib/structs/AuxStructs.h>
#include <taLib/structs/VectorMatrix.h>
#include <taLib/structs/Functions.h>
#include <taLib/structs/Lists2.h>
// #include <taLib/structs/ClassifierData.h>

// ////////////////////////////// mlpack stuff//////////////////////
// #include <taLib/my_mlpack/core.hpp>
// #include <taLib/my_mlpack/fastmks/fastmks_stat.hpp>
// #include <taLib/my_mlpack/fastmks/fastmks.hpp>
// #include <taLib/my_mlpack/fastmks/fastmks_impl.hpp>
// #include <taLib/my_mlpack/fastmks/fastmks_rules.hpp>
// #include <taLib/my_mlpack/fastmks/fastmks_rules_impl.hpp>
// ///////////////////////////////////////////////////////////////////
// /////////// l2ap stuff ///////////////
// #include <taLib/ap/includes.h>
// #include <taLib/ap/connect.h>
// #include <taLib/structs/l2apIndex.h>
// #include <taLib/structs/LshIndex2.h>
// #include <taLib/structs/BlshIndex2.h>
/////////////////////////////////////


// #include <taLib/structs/TreeIndex.h>
#include <taLib/structs/Candidates.h>
#include <taLib/structs/TAState.h>
#include <taLib/structs/RetrievalArguments.h>
#include <taLib/structs/QueryBatch.h>
#include <taLib/structs/ProbeBucket.h>
#include <taLib/structs/Bucketize.h>
#include <taLib/structs/CandidateVerification.h>
// #include <taLib/structs/Cluster.h>


#include <taLib/retrieval/Retriever.h>
#include <taLib/retrieval/ListsTuneData.h>
#include <taLib/retrieval/TuneTopk.h>
#include <taLib/retrieval/coord.h>
#include <taLib/retrieval/incr2.h>
#include <taLib/retrieval/ta.h>
#include <taLib/retrieval/mixed.h>
// #include <taLib/retrieval/singleTree.h>
// #include <taLib/retrieval/apRetriever.h>
// #include <taLib/retrieval/lshRetriever2.h>
// #include <taLib/retrieval/blshRetriever2.h>


// #include <taLib/PCATree/PCATree.h>

#include <taLib/algos/naiveAlgo.h>
#include <taLib/algos/TA_all.h>
// #include <taLib/algos/LSH_all.h>
#include <taLib/algos/algo_with_tuning.h>


#endif