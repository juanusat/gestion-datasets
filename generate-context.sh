#!/bin/bash

OUTPUT_FILE=out-contexto.txt

touch $OUTPUT_FILE
truncate -s 0 $OUTPUT_FILE

echo "listado" >> $OUTPUT_FILE
ls -1F --color=never >> $OUTPUT_FILE

tree origins > tree-full.txt
tree -d origins > tree-dirs.txt
