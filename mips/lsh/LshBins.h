/* 
 * File:   LshBins2.h
 * Author: chteflio
 *
 * Created on October 8, 2015, 9:54 AM
 */

#ifndef LSHBINS_H
#define	LSHBINS_H
#include <boost/dynamic_bitset.hpp>

namespace mips {

    class LshBins {
    protected:
        long long userOffset;
        row_type bytesPerCode;
        row_type numHashTables;
        row_type nVectors;
        long long numBinsPerHashTable;

        inline uint32_t popcount(uint32_t v) {
            // http://graphics.stanford.edu/~seander/bithacks.html
            v = v - ((v >> 1) & 0x55555555); // reuse input as temporary
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333); // temp
            uint32_t c = ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24; // count

            return c;
        }

        inline bool processPair(row_type hashGroups, int32_t *minMatches, uint32_t *x, uint32_t *y) {

            int matches = 0;



            for (row_type i = 0; i < hashGroups; i++) {
                matches += 32 - __builtin_popcount(*(x++) ^ *(y++)); //__builtin_popcount

                //                std::cout<<matches<<" vs "<< minMatches[i]<<std::endl;

                if (matches < minMatches[i]) {
                    return false;
                }
            }

            return true;

        }

    public:


        inline LshBins() = default;

        inline void init(int bytesPerCode_, int numHashTables_, row_type nVectors_) {
            bytesPerCode = bytesPerCode_;
            numHashTables = numHashTables_;
            nVectors = nVectors_;
            userOffset = bytesPerCode * numHashTables;
            numBinsPerHashTable = pow(2, bytesPerCode * 8);
        }

        inline virtual void printBins() {
        }

        inline virtual void resizeBins(row_type size) {
        }

        inline virtual void populateBins(uint8_t* sketch, row_type startBlock, row_type endBlock,
                std::vector<row_type>& countsOfBlockValues, bool forBlsh) {
        }

        inline virtual void getCandidates(uint8_t* querySketches, row_type queryPos, row_type* candidatesToVerify, row_type& numCandidatesToVerify,
                boost::dynamic_bitset<>& done, row_type activeBlocks, row_type probeBucketStartPos) {
        }

        inline virtual void getCandidatesWithBlshPruning(uint8_t* querySketches, row_type queryPos, uint8_t* probeSketches, int32_t *minMatches, row_type* candidatesToVerify, row_type& numCandidatesToVerify,
                boost::dynamic_bitset<>& done, row_type activeBlocks, row_type probeBucketStartPos) {
        }


    };

    // only for 8bit signatures

    class LshBinsDense : public LshBins {
    public:
        std::vector<row_type > binsOffsets;
        std::vector<row_type> data;

        inline LshBinsDense() = default;

        inline virtual void printBins() {
        }

        inline virtual void resizeBins(row_type size) {
        }

        inline void populateBins(uint8_t* sketch, row_type startBlock, row_type endBlock,
                std::vector<row_type>& countsOfBlockValues, bool forBlsh) {

            data.resize(endBlock * nVectors);
            row_type numValuesPerBlock = 256;
            binsOffsets.resize(endBlock * (numValuesPerBlock + 1));

            // update hashBuckets
            for (int block = startBlock; block < endBlock; ++block) {
                row_type startOffset = bytesPerCode * block;
                int blockOffsetOnData = block * nVectors;

                for (row_type i = 0; i < nVectors; ++i) {
                    long long t = i * userOffset + startOffset;
                    uint8_t key = sketch[t];
                    countsOfBlockValues[key]++;
                }

                binsOffsets[block * (numValuesPerBlock + 1) ] = 0;
                row_type startOffset2 = block * (numValuesPerBlock + 1);
                for (int value = 1; value <= numValuesPerBlock; ++value) {
                    binsOffsets[startOffset2 + value] = binsOffsets[startOffset2 + value - 1] + countsOfBlockValues[value - 1];
                }

                if (forBlsh) {
                    for (row_type i = 0; i < nVectors; ++i) {
                        long long t = i * userOffset + startOffset;
                        uint8_t key = sketch[t];
                        //                    sketch[t] = 0;
                        int j = binsOffsets[startOffset2 + key + 1] - countsOfBlockValues[key];
                        countsOfBlockValues[key]--;
                        data[blockOffsetOnData + j] = i;
                    }

                } else {
                    for (row_type i = 0; i < nVectors; ++i) {
                        long long t = i * userOffset + startOffset;
                        uint8_t key = sketch[t];
                        sketch[t] = 0;
                        int j = binsOffsets[startOffset2 + key + 1] - countsOfBlockValues[key];
                        countsOfBlockValues[key]--;
                        data[blockOffsetOnData + j] = i;
                    }

                }

            }

        }

        // lsh version

        inline void getCandidates(uint8_t* querySketches, row_type queryPos, row_type* candidatesToVerify, row_type& numCandidatesToVerify,
                boost::dynamic_bitset<>& done, row_type activeBlocks, row_type probeBucketStartPos) {

            done.reset();
            long long t = queryPos * userOffset;
            row_type offset2 = 256 + 1;
            row_type offset = 0;


            for (row_type block = 0; block < activeBlocks; ++block) {

                uint8_t key = querySketches[t];
                t++;

                row_type start = binsOffsets[block * offset2 + key];
                row_type end = binsOffsets[block * offset2 + key + 1];

                for (row_type j = start; j < end; ++j) { // find all with same hash value in probe

                    row_type uId = data[offset + j];

                    if (done[uId])
                        continue;

                    done[uId] = true;



                    candidatesToVerify[numCandidatesToVerify] = uId + probeBucketStartPos;
                    numCandidatesToVerify++;
                }
                offset += nVectors;

            }
        }

        // blsh version

        inline void getCandidatesWithBlshPruning(uint8_t* querySketches, row_type queryPos, uint8_t* probeSketches, int32_t *minMatches, row_type* candidatesToVerify, row_type& numCandidatesToVerify,
                boost::dynamic_bitset<>& done, row_type activeBlocks, row_type probeBucketStartPos) {

            done.reset();
            long long t = queryPos * userOffset;
            row_type offset2 = 256 + 1;
            row_type offset = 0;

            uint32_t * x = (uint32_t *) (&querySketches[t]);


            int hashGroups = activeBlocks * bytesPerCode / 4;


            //            std::cout<<"activeBlocks: "<<activeBlocks<<" hashGroups: "<<hashGroups<<std::endl;

            for (row_type block = 0; block < activeBlocks; ++block) {

                uint8_t key = querySketches[t];
                t++;

                row_type start = binsOffsets[block * offset2 + key];
                row_type end = binsOffsets[block * offset2 + key + 1];

                for (row_type j = start; j < end; ++j) { // find all with same hash value in probe

                    row_type uId = data[offset + j];

                    if (done[uId])
                        continue;

                    done[uId] = true;

                    //                    std::cout<<"position in probe sketches: "<<uId * userOffset<<std::endl;
                    //                    std::cout<<"userOffset: "<<userOffset<<std::endl;


                    uint32_t * y = (uint32_t *) (&(probeSketches[uId * userOffset]));

                    //                    std::cout<<"x: "<<*x<< " vs y: "<<*y<<std::endl;

                    bool flag = processPair(hashGroups, minMatches, x, y);

                    if (flag) {
                        candidatesToVerify[numCandidatesToVerify] = uId + probeBucketStartPos;
                        numCandidatesToVerify++;
                    }

                }
                offset += nVectors;

            }
        }
    };

    template<typename T>// for 16, 32 64 bit signatures
    class LshBinsSparse : public LshBins {
    public:

        std::vector<unordered_map<T, std::vector<row_type> > > bins;
        inline LshBinsSparse() = default;

        inline void populateBins(uint8_t* sketch, row_type startBlock, row_type endBlock,
                std::vector<row_type>& countsOfBlockValues, bool forBlsh) {
            bins.resize(endBlock);
            row_type startOffset = bytesPerCode * startBlock;


            for (row_type i = 0; i < nVectors; ++i) {
                long long t = i * userOffset + startOffset;

                if (forBlsh) {
                    for (int block = startBlock; block < endBlock; ++block) {
                        T key = 0;
                        memcpy(&key, sketch + t, bytesPerCode);

                        t += bytesPerCode;
                        bins[block][key].push_back(i);
                    }
                } else {
                    for (int block = startBlock; block < endBlock; ++block) {
                        T key = 0;
                        memcpy(&key, sketch + t, bytesPerCode);
                        memset(sketch + t, 0, bytesPerCode);

                        t += bytesPerCode;
                        bins[block][key].push_back(i);
                    }
                }




            }
        }

        inline virtual void printBins() {
            for (int block = 0; block < bins.size(); ++block) {

                std::cout << "block: " << block << std::endl;
                for (auto it = bins[block].begin(); it != bins[block].end(); it++) {
                    std::cout << "hv: " << (int) it->first << " size: " << it->second.size() << std::endl;
                }
            }
        }

        inline virtual void resizeBins(row_type size) {
            bins.resize(size);
        }

        inline void getCandidates(uint8_t* querySketches, row_type queryPos, row_type* candidatesToVerify, row_type& numCandidatesToVerify,
                boost::dynamic_bitset<>& done, row_type activeBlocks, row_type probeBucketStartPos) {


            done.reset();
            long long t = queryPos * userOffset;

            for (row_type block = 0; block < activeBlocks; ++block) {

                T key = 0;
                memcpy(&key, querySketches + t, bytesPerCode);
                t += bytesPerCode;

                auto it = bins[block].find(key);

                if (it != bins[block].end()) {

                    for (row_type i = 0; i < it->second.size(); ++i) {
                        row_type uId = it->second.at(i);

                        if (done[uId])
                            continue;

                        done[uId] = true;

                        candidatesToVerify[numCandidatesToVerify] = uId + probeBucketStartPos;
                        numCandidatesToVerify++;
                    }
                }
            }
        }

        inline void getCandidatesWithBlshPruning(uint8_t* querySketches, row_type queryPos, uint8_t* probeSketches, int32_t *minMatches, row_type* candidatesToVerify, row_type& numCandidatesToVerify,
                boost::dynamic_bitset<>& done, row_type activeBlocks, row_type probeBucketStartPos) {


            done.reset();
            long long t = queryPos * userOffset;

            uint32_t * x = (uint32_t *) (&querySketches[t]);
            int hashGroups = activeBlocks * bytesPerCode / 4;

            for (row_type block = 0; block < activeBlocks; ++block) {

                T key = 0;
                memcpy(&key, querySketches + t, bytesPerCode);
                t += bytesPerCode;

                auto it = bins[block].find(key);

                if (it != bins[block].end()) {

                    for (row_type i = 0; i < it->second.size(); ++i) {
                        row_type uId = it->second.at(i);

                        if (done[uId])
                            continue;

                        done[uId] = true;

                        uint32_t * y = (uint32_t *) (&(probeSketches[uId * numHashTables]));

                        bool flag = processPair(hashGroups, minMatches, x, y);

                        if (flag) {
                            candidatesToVerify[numCandidatesToVerify] = uId + probeBucketStartPos;
                            numCandidatesToVerify++;
                        }

                    }
                }
            }
        }


    };



}


#endif	/* LSHBINS_H */

