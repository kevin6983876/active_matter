#!/bin/bash

gcc gMAM_Schlogl_2ndOrder.c -lfftw3 -lm -O3


zeta=0.5 #0.5 for Crank-Nicolson solver

Ncopy=100
Nx=40
dt=0.01

plotStep=20000
iterations=60000

for upward in 1;
do  	
    for D in  11;
    do
	if [ $upward -gt 0 ]
	then
	    dir="upward_Ncopy${Ncopy}L${Nx}D${D}"
	else
	    dir="downward_Ncopy${Ncopy}L${Nx}D${D}"
	fi
	mkdir "$dir"
	cp ./a.out  "$dir"
	cp ./*.py "$dir"
	cd "$dir"

	#./a.out Ncopy Nx dt zeta plotStep iterations D up_boolean
        ./a.out $Ncopy $Nx $dt $zeta  $plotStep $iterations $D $upward
	
	cd ..
    done

done
