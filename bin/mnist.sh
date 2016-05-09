#!/bin/bash

#java -agentlib:hprof=file=hprof.txt,cpu=times -classpath lib/commons-math3-3.5.jar:build/classes org.singularity.application.mnist.MnistReader $1 $2
java -classpath lib/commons-math3-3.5.jar:build/classes org.singularity.application.mnist.MnistReader $1 $2
