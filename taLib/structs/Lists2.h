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
 * File:   Lists2.h
 * Author: chteflio
 *
 * Created on January 5, 2015, 3:59 PM
 */

#ifndef LISTS2_H
#define	LISTS2_H


namespace ta {

    class Index {
    public:
        bool initialized;
        omp_lock_t writelock;

        inline Index() : initialized(false) {
            omp_init_lock(&writelock);
        }

        inline ~Index() {
            omp_destroy_lock(&writelock);
        }

    };

    class QueueElementLists : public Index {

        /*
         * returns true if there is possibility for sufficient, otherwise false
         */
        inline bool getBounds(double qi, double theta, col_type col, std::pair<row_type, row_type>& necessaryIndices) {

            bool suff = false;
            double base, x, root1, root2;
            std::pair<double, double> necessaryValues;
            std::vector<QueueElement>::iterator it;

            // get bounds in the form of values
            base = theta * qi;
            x = sqrt((theta * theta - 1) * (qi * qi - 1));

            root1 = base + x;
            root2 = base - x;

            necessaryValues.first = root2;
            necessaryValues.second = root1;

            row_type start = col * size;
            row_type end = (col + 1) * size;

            if (qi > 0 && necessaryValues.second >= theta / qi) {
                necessaryValues.second = 1;
                suff = true;

            } else if (qi < 0 && necessaryValues.first <= theta / qi) {
                necessaryValues.first = -1;
                suff = true;
            }

            if (necessaryValues.first <= sortedCoord[col * size].data) {
                necessaryIndices.first = start;
            } else {

                it = std::lower_bound(sortedCoord.begin() + start, sortedCoord.begin() + end, QueueElement(necessaryValues.first, 0));
                necessaryIndices.first = (it - sortedCoord.begin());
            }

            if (necessaryValues.second > sortedCoord[(col + 1) * size - 1].data) {
                necessaryIndices.second = end;
            } else {
                it = std::upper_bound(sortedCoord.begin() + start, sortedCoord.begin() + end, QueueElement(necessaryValues.second, 0));
                necessaryIndices.second = (it - sortedCoord.begin());
            }


            return suff;
        }

    public:


        std::vector<QueueElement> sortedCoord;
        col_type colNum;
        row_type size;

        inline QueueElementLists() {
        }

        inline ~QueueElementLists() {

        }

        inline void initializeLists(const VectorMatrix& matrix, ta_size_type start = 0, ta_size_type end = 0) {
            omp_set_lock(&writelock);

            if (!initialized) {

                colNum = matrix.colNum;

                if (start == end) {
                    start = 0;
                    end = matrix.rowNum;
                }
                size = end - start;
                sortedCoord.reserve(colNum * size);

                for (col_type j = 0; j < colNum; j++) {
                    for (row_type i = start; i < end; i++) { // scans the matrix as it is, i.e., perhaps in sorted order

                        sortedCoord.push_back(QueueElement(matrix.getMatrixRowPtr(i)[j], i - start));
                        // QueueElement.id is the position of the vector in the matrix, not necessarily the vectorID
                    }
                    std::sort(sortedCoord.begin()+(j * size), sortedCoord.end(), std::less<QueueElement>());
                }
                initialized = true;
            }
            omp_unset_lock(&writelock);
        }

        inline row_type getRowPointer(row_type row, col_type col) {
            return sortedCoord[col * size + row].id;
        }

        inline QueueElement* getElement(row_type pos) {
            return &sortedCoord[pos];
        }

        inline double getValue(row_type row, col_type col) {
            return sortedCoord[col * size + row].data;
        }

        inline col_type getColNum() {
            return colNum;
        }

        inline row_type getRowNum() {
            return size;
        }

        inline bool calculateIntervals(const double* query, const col_type* listsQueue, std::vector<IntervalElement>& intervals,
                double localTheta, col_type lists) {

            std::pair<row_type, row_type> necessaryIndices;

            for (col_type i = 0; i < lists; i++) {

                getBounds(query[listsQueue[i]], localTheta, listsQueue[i], necessaryIndices);

                intervals[i].col = listsQueue[i];
                intervals[i].start = necessaryIndices.first;
                intervals[i].end = necessaryIndices.second;

                if (intervals[i].end <= intervals[i].start) {
                    return false;
                }
            }

            std::sort(intervals.begin(), intervals.begin() + lists);
            return true;
        }


    };

    // contains the sorted lists
    // 1st Dimension: coordinates  2nd Dimension: rows (row pointers to the NormMatrix)

    class IntLists : public Index {
        std::vector<double> values;

        inline void getBounds(double qi, double theta, col_type col, std::pair<row_type, row_type>& necessaryIndices) {

            double base, x, root1, root2;
            std::pair<double, double> necessaryValues;
            std::vector<double>::iterator it;

            // get bounds in the form of values
            base = theta * qi;
            x = sqrt((theta * theta - 1) * (qi * qi - 1));

            root1 = base + x;
            root2 = base - x;

            necessaryValues.first = root2;
            necessaryValues.second = root1;

            row_type start = col * size;
            row_type end = (col + 1) * size;



            if (qi > 0) {
                necessaryValues.second = (necessaryValues.second >= theta / qi ? 1 : necessaryValues.second);
            } else if (qi < 0) {
                necessaryValues.first = (necessaryValues.first <= theta / qi ? -1 : necessaryValues.first);
            }

            if (necessaryValues.first <= values[start]) {
                necessaryIndices.first = start;
            } else {
                it = std::lower_bound(values.begin() + start, values.begin() + end, necessaryValues.first);
                necessaryIndices.first = it - values.begin();
            }

            if (necessaryValues.second > values[end - 1]) {
                necessaryIndices.second = end;
            } else {
                it = std::upper_bound(values.begin() + start, values.begin() + end, necessaryValues.second);
                necessaryIndices.second = it - values.begin();
            }
            
            
        }

    public:
        col_type colNum;
        row_type size;

        std::vector<row_type> ids;

        inline IntLists() {
        }

        inline ~IntLists() {
        }

        inline void initializeLists(const VectorMatrix& matrix, ta_size_type start = 0, ta_size_type end = 0) {
            omp_set_lock(&writelock);
            if (!initialized) {
                std::vector<std::vector<QueueElement> > sortedCoord;

                sortedCoord.clear();
                sortedCoord.resize(matrix.colNum);

                colNum = matrix.colNum;


                if (start == end) {
                    start = 0;
                    end = matrix.rowNum;
                }
                size = end - start;

                ids.reserve(colNum * size);
                values.reserve(colNum * size);


                //                for (row_type i = start; i < end; i++) { // scans the matrix as it is, i.e., perhaps in sorted order
                //                    for (col_type j = 0; j < colNum; j++) {
                //                        sortedCoord[j].push_back(QueueElement(matrix.getMatrixRowPtr(i)[j], i - start)); // i is the position of the vector in the matrix, not necessarily the vectorID
                //                    }
                //                }
                //                for (col_type j = 0; j < colNum; j++) {
                //                    std::sort(sortedCoord[j].begin(), sortedCoord[j].end(), std::less<QueueElement>());
                //                }

                for (col_type i = 0; i < colNum; i++) {

                    for (row_type j = start; j < end; j++) { // scans the matrix as it is, i.e., perhaps in sorted order
                        sortedCoord[i].push_back(QueueElement(matrix.getMatrixRowPtr(j)[i], j - start));
                        // i is the position of the vector in the matrix, not necessarily the vectorID
                    }

                    std::sort(sortedCoord[i].begin(), sortedCoord[i].end(), std::less<QueueElement>());

                    for (row_type j = 0; j < sortedCoord[i].size(); j++) {
                        ids.push_back(sortedCoord[i][j].id);
                        values.push_back(sortedCoord[i][j].data);
                    }
                }
                initialized = true;
            }
            omp_unset_lock(&writelock);
        }

        //        inline void initializeLists(std::vector<std::vector<QueueElement> >& sortedCoord) { // initialize directly from an Incr bucket
        //            col_type colNum = sortedCoord.size();
        //
        //            ids.resize(colNum);
        //            values.resize(colNum);
        //
        //            for (col_type i = 0; i < colNum; i++) {
        //                for (row_type j = 0; j < sortedCoord[i].size(); j++) {
        //                    ids[i].push_back(sortedCoord[i][j].id);
        //                    values[i].push_back(sortedCoord[i][j].data);
        //                }
        //            }
        //
        //            initialized = true;
        //        }

        
        inline row_type getRowPointer(row_type row, col_type col) {
            return ids[col * size + row];
        }

        inline row_type* getElement(row_type pos) {
            return &ids[pos];
        }

        inline double getValue(row_type row, col_type col) {
            return values[col * size + row];
        }

        inline col_type getColNum() {
            return colNum;
        }

        inline row_type getRowNum() {
            return size;
        }

        inline bool calculateIntervals(const double* query, const col_type* listsQueue, std::vector<IntervalElement>& intervals,
                double localTheta, col_type lists) {

            std::pair<row_type, row_type> necessaryIndices;

            for (col_type i = 0; i < lists; i++) {

                getBounds(query[listsQueue[i]], localTheta, listsQueue[i], necessaryIndices);

                intervals[i].col = listsQueue[i];
                intervals[i].start = necessaryIndices.first;
                intervals[i].end = necessaryIndices.second;

                if (intervals[i].end <= intervals[i].start) {
                    return false;
                }
            }

            std::sort(intervals.begin(), intervals.begin() + lists);

            return true;
        }

    };

    
//     class IntLists : public Index {
//        std::vector<std::vector<double> > values;
//
//        inline void getBounds(double qi, double theta, col_type col, std::pair<row_type, row_type>& necessaryIndices) {
//
//            double base, x, root1, root2;
//            std::pair<double, double> necessaryValues;
//            std::vector<double>::iterator low, up;
//
//            // get bounds in the form of values
//            base = theta * qi;
//            x = sqrt((theta * theta - 1) * (qi * qi - 1));
//
//            root1 = base + x;
//            root2 = base - x;
//
//            necessaryValues.first = root2;
//            necessaryValues.second = root1;
//
//            if (qi > 0) {
//                necessaryValues.second = (necessaryValues.second >= theta / qi ? 1 : necessaryValues.second);
//            } else if (qi < 0) {
//                necessaryValues.first = (necessaryValues.first <= theta / qi ? -1 : necessaryValues.first);
//            }
//
//            if (necessaryValues.first <= values[col][0]) {
//                necessaryIndices.first = 0;
//            } else {
//                low = std::lower_bound(values[col].begin(), values[col].end(), necessaryValues.first);
//                necessaryIndices.first = low - values[col].begin();
//            }
//
//            if (necessaryValues.second > values[col][values.size() - 1]) {
//                necessaryIndices.second = values[col].size();
//            } else {
//                //up = std::upper_bound (values[col].begin()+necessaryIndices.first, values[col].end(), necessaryValues.second);
//                up = std::upper_bound(values[col].begin(), values[col].end(), necessaryValues.second);
//                necessaryIndices.second = up - values[col].begin();
//            }
//
//
//        }
//
//    public:
//
//        std::vector<std::vector<row_type> > ids;
//
//        inline IntLists() {}
//
//        inline ~IntLists() {}
//
//        inline void initializeLists(const VectorMatrix& matrix, ta_size_type start = 0, ta_size_type end = 0) {
//            omp_set_lock(&writelock);
//            if (!initialized) {
//                std::vector<std::vector<QueueElement> > sortedCoord;
//
//                sortedCoord.clear();
//                sortedCoord.resize(matrix.colNum);
//
//                col_type colNum = matrix.colNum;
//
//
//                if (start == end) {
//                    start = 0;
//                    end = matrix.rowNum;
//                }
//
//                ids.resize(colNum);
//                values.resize(colNum);
//
//
//                for (row_type i = start; i < end; i++) { // scans the matrix as it is, i.e., perhaps in sorted order
//                    for (col_type j = 0; j < colNum; j++) {
//                        sortedCoord[j].push_back(QueueElement(matrix.getMatrixRowPtr(i)[j], i - start)); // i is the position of the vector in the matrix, not necessarily the vectorID
//                    }
//                }
//                for (col_type j = 0; j < colNum; j++) {
//                    std::sort(sortedCoord[j].begin(), sortedCoord[j].end(), std::less<QueueElement>());
//                }
//
//                for (col_type i = 0; i < colNum; i++) {
//                    for (row_type j = 0; j < sortedCoord[i].size(); j++) {
//                        ids[i].push_back(sortedCoord[i][j].id);
//                        values[i].push_back(sortedCoord[i][j].data);
//                    }
//                }
//                initialized = true;
//            }
//            omp_unset_lock(&writelock);
//        }
//
//        inline void initializeLists(std::vector<std::vector<QueueElement> >& sortedCoord) { // initialize directly from an Incr bucket
//            col_type colNum = sortedCoord.size();
//
//            ids.resize(colNum);
//            values.resize(colNum);
//
//            for (col_type i = 0; i < colNum; i++) {
//                for (row_type j = 0; j < sortedCoord[i].size(); j++) {
//                    ids[i].push_back(sortedCoord[i][j].id);
//                    values[i].push_back(sortedCoord[i][j].data);
//                }
//            }
//
//            initialized = true;
//        }
//
//        inline row_type getRowPointer(row_type row, col_type col) {
//            return ids[col][row];
//        }
//
//        inline double getValue(row_type row, col_type col) {
//            return values[col][row];
//        }
//
//        inline col_type getColNum() {
//            return ids.size();
//        }
//
//        inline row_type getRowNum() {
//            return ids[0].size();
//        }
//        
//        inline row_type* getElement(row_type pos) {//dummy
//            return &ids[pos][0];
//        }
//
//        inline bool calculateIntervals(const double* query, const col_type* listsQueue, std::vector<IntervalElement>& intervals,
//                double localTheta, col_type lists) {
//
//            std::pair<row_type, row_type> necessaryIndices;
//
//            for (col_type i = 0; i < lists; i++) {
//
//                getBounds(query[listsQueue[i]], localTheta, listsQueue[i], necessaryIndices);
//
//                intervals[i].col = listsQueue[i];
//                intervals[i].start = necessaryIndices.first;
//                intervals[i].end = necessaryIndices.second;
//
//                if (intervals[i].end <= intervals[i].start) {
//                    return false;
//                }
//            }
//
//            std::sort(intervals.begin(), intervals.begin() + lists);
//
//            return true;
//        }
//
//    };

}
#endif	/* LISTS2_H */
