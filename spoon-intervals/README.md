# How to compile:

mvn clean compile assembly:single

# How to run

java -cp this-proj.jar Main -d path-to-java-source -b buggy-lines intervals.json
