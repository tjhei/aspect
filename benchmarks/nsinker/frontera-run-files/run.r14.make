
nodes=4
tt=56
aa=-1

for nodes in "1" "2" "4" "8" "16" "32" "64" "128" "256";
do
tasks=$(($nodes * $tt))
echo "nodes=$nodes tpn=$tt tasks=$tasks"
#JJ="-j"
aa=$(($aa + 1))
sed "s/&NODES&/$nodes/g" run.r14.base | sed "s/&TASKS&/$tasks/g" | sed "s/&ADAPTIVE&/$aa/g" > run.r14.$nodes

done
