#!/usr/bin/env bash
wget 'http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/bikes.tar.gz' && /
mkdir bikes && /
tar -xzf bikes.tar.gz -C bikes && /
rm bikes.tar.gz

wget 'http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz' && /
mkdir graf && /
tar -xzf trees.tar.gz -C graf && /
rm graf.tar.gz

wget 'http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/trees.tar.gz' && /
mkdir trees && /
tar -xzf trees.tar.gz -C trees && /
rm trees.tar.gz

wget 'http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/wall.tar.gz' && /
mkdir wall && /
tar -xzf wall.tar.gz -C wall && /
rm wall.tar.gz

wget 'http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/bark.tar.gz' && /
mkdir bark && /
tar -xzf bark.tar.gz -C bark && /
rm bark.tar.gz

wget 'http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/boat.tar.gz' && /
mkdir boat && /
tar -xzf boat.tar.gz -C boat && /
rm boat.tar.gz

wget 'http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/leuven.tar.gz' && /
mkdir leuven && /
tar -xzf leuven.tar.gz -C leuven && /
rm leuven.tar.gz

wget 'http://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/ubc.tar.gz' && /
mkdir ubc && /
tar -xzf ubc.tar.gz -C ubc && /
rm ubc.tar.gz

printf "1  0  0\n0  1  0\n0  0  1" > "ubc/H1to2p"
printf "1  0  0\n0  1  0\n0  0  1" > "ubc/H1to3p"
printf "1  0  0\n0  1  0\n0  0  1" > "ubc/H1to4p"
printf "1  0  0\n0  1  0\n0  0  1" > "ubc/H1to5p"
printf "1  0  0\n0  1  0\n0  0  1" > "ubc/H1to6p"

rm *.tar.gz