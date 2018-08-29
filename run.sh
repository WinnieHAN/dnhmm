cd ~/Code/wsj
rm *.t7
cd ~/Code/neuralHMM
th main.lua -datapath ../wsj -nstates 45 -niters 20  -hidsize 512 -mnbz 128  -nloops 6 -cuda -maxlen 81  -nlayers 3 -modelpath ../save -model lstmconv
