### K-means improvement report

Most long function was `computeAssignments`. So i decided improve this function

Elapsed time of original implementation: 1.44 seconds
- I change iterations - in inner loop we iterate over all points of dataset, and this give huge load on memory subsystem - 
loading and invalidating cache in each iteration. This gave ~1.14 speedup in comparison with original implementation;
- Next step is removing initialization of `minDist` array. Initialize this array by distance to first centroid. This gave ~1.038 speedup in comparison with previous implementation;
- Next step - parallelization via OpenMP - give improve around ~6.43 with 8 cores (processor - AMD Ryzen 7 4800H with Radeon Graphics)
- Next attempt - inlining `dist` function and reorganization of inner loop by centroids and dimension. Inlining hepls by removing `sqrt` in each distance computation. Only in last assignment `sqrt` will be calling.
But this optimization don't give speedup in comparison with previous implementation;


