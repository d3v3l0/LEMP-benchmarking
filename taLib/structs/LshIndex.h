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

/* 
 * File:   LshIndex.h
 * Author: chteflio
 *
 * Created on June 26, 2015, 2:33 PM
 */

#ifndef LSHINDEX_H
#define	LSHINDEX_H


namespace ta {


    

    class LshIndex : public Index {
        CosineSketches* cosSketches;
        LshBins* lshBins;
        std::vector<row_type> initializedSketches;
        row_type nVectors;
        
    public:       
        
        row_type initializedSketchesForIndex;

        inline LshIndex() : initializedSketchesForIndex(0), cosSketches(nullptr), lshBins(nullptr) {
        }

        inline ~LshIndex() {
            if (cosSketches != nullptr)
                delete cosSketches;

            if (lshBins != nullptr)
                delete lshBins;

        }

        // call from the probe bucket

        inline void checkAndReallocateAll(const VectorMatrix* matrix, bool forProbe, row_type start, row_type end, row_type activeBuckets,
                std::vector<float>& sums, std::vector<row_type>& countsOfBlockValues, uint8_t* my_sketches) {

            row_type startBlock = initializedSketchesForIndex;
            uint8_t* ptrSketches = (forProbe ? my_sketches : cosSketches->sketches);

            if (startBlock < activeBuckets) { // reallocate

                // find end block
                row_type endBlock = activeBuckets;
                cosSketches->buildBatch(matrix, start, end, sums, ptrSketches, startBlock, endBlock);

                if (forProbe) {
                    lshBins->populateBins(ptrSketches, startBlock, endBlock, countsOfBlockValues);
                }
                initializedSketchesForIndex = endBlock;
            }
        }

        // call for query

        inline void checkAndReallocateSingle(const VectorMatrix* matrix, row_type posInMatrix, row_type posInBucket, row_type activeBuckets,
                std::vector<float>& sums) {
            row_type startBlock = initializedSketches[posInBucket];
            if (startBlock < activeBuckets) { // reallocate

                // find end block
                row_type endBlock = activeBuckets;
                double* vec = matrix->getMatrixRowPtr(posInMatrix);

                row_type startOffset = coreLshInfo.bytesPerCode * startBlock;
                row_type startHashBit = startBlock * coreLshInfo.hashCodeLength;
                row_type endHashBit = endBlock * coreLshInfo.hashCodeLength;

                cosSketches->buildSingle(vec, posInBucket, matrix->colNum, sums, cosSketches->sketches, startBlock, endBlock, startHashBit, endHashBit, startOffset);
                initializedSketches[posInBucket] = endBlock;

            }
        }

        inline void getCandidates(uint8_t* querySketches, row_type queryPos, row_type* candidatesToVerify, row_type& numCandidatesToVerify,
                boost::dynamic_bitset<>& done, row_type activeBlocks, row_type probeBucketStartPos) {
            lshBins->getCandidates(querySketches, queryPos, candidatesToVerify, numCandidatesToVerify, done, activeBlocks, probeBucketStartPos);
        }
        
        inline uint8_t* getSketch(){
            return cosSketches->sketches;
        }

        /* Both for queries and probe vectors
         */
        inline void initializeLists(const VectorMatrix& matrix, bool forProbeVectors, ta_size_type start = 0, ta_size_type end = 0) {

            omp_set_lock(&writelock);

            if (!initialized) {

                end = (end == 0 ? matrix.rowNum : end);
                nVectors = end - start;
                cosSketches = new CosineSketches();

                if (!forProbeVectors) {
                    initializedSketches.resize(nVectors, 0);
                    cosSketches->alloc(nVectors);
                } else {

                    switch (LSH_CODE_LENGTH) {
                        case 8:
                            lshBins = new LshBinsDense();
                            break;
                        case 16:
                            lshBins = new LshBinsSparse<uint16_t>();
                            break;
                        case 24:
                        case 32:
                            lshBins = new LshBinsSparse<uint32_t>();
                            break;
                        default:
                            lshBins = new LshBinsSparse<uint64_t>();
                            break;
                    }

                    lshBins->init(nVectors);
                }
                initialized = true;

            }
            omp_unset_lock(&writelock);
        }



        // for runLSH_all queries

        inline void initializeLists(const VectorMatrix& matrix) {

            omp_set_lock(&writelock);

            if (!initialized) {
                nVectors = matrix.rowNum;
                cosSketches = new CosineSketches();
                initialized = true;
            }
            omp_unset_lock(&writelock);
        }

        /*
         * this is for calling while tuning and only for the queries. No need to lock
         */
        inline void initializeLists(row_type sampleSize) {
            initializedSketches.resize(sampleSize, 0);
            cosSketches = new CosineSketches();
            cosSketches->alloc(sampleSize);
            initialized = true;
        }

    };





}


#endif	/* LSHINDEX_H */

