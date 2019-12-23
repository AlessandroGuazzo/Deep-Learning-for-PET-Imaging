#!/bin/bash

echo File to process:

read inputName

echo Processing $inputname

echo Folder Name:

read folder

mkdir $folder

echo Ground Truth Reconstruction

lr5gen -sum $inputName -clobber GT.3dlor.lr5 -fenorm /site/minipet/etc/normtable4MBq_180130-171730.3D.lr5
lr5bin -clobber GT.3dlor.lr5 GT.2dlor.lr5
rec -v5 -thread 8 GT.2dlor.lr5 GT.mnc

mv GT.mnc $folder

counter=0

while [ $counter -le 60 ]

do

echo Processing shot $counter

lr5gen -sum -rshot 0 $counter -clobber $inputName temp.3dlor.lr5 -fenorm /site/minipet/etc/normtable4MBq_180130-171730.3D.lr5
lr5bin -clobber temp.3dlor.lr5 temp.2dlor.lr5
lr5bin -sin -mnc temp.2dlor.lr5 sino${counter}min.sino.mnc
mv sino${counter}min.sino.mnc $folder

if [ $counter -gt 44 ]

then

lr5gen -shot $counter -clobber $inputName temp.3dlor.lr5 -fenorm /site/minipet/etc/normtable4MBq_180130-171730.3D.lr5
lr5bin -clobber temp.3dlor.lr5 temp.2dlor.lr5
lr5bin -sin -mnc temp.2dlor.lr5 sino${counter}x1min.sino.mnc
mv sino${counter}x1min.sino.mnc $folder

fi

((counter++))

done

echo Processing Completed!!!
