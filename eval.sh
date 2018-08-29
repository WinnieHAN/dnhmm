th main.lua -datapath ../wsj -nstates 45 -hidsize 512 -nlayers 3 -cuda -model ../save/lstmconv.iter19.t7  -input ../wsj_all/WSJ_s23_tree_dep_words_0  -output ../tagged_file.txt
python2 eval.py ../tagged_file.txt ../wsj_all/WSJ_s23_tree_tags
