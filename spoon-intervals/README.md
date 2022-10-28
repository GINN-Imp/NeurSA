# How to compile:

mvn clean compile assembly:single

# How to run

java -cp this-proj.jar Main path-to-java-source buggy-lines intervals.json

# generate data for caching

##Both buggy and fixed

python generateDataset.py --cacheFile 2 

##generate test data

time python generateDataset.py extractedFiles --useCached 2 --isDumpCFG 1; time python generateDataset.py extractedFiles --useCached 2;
