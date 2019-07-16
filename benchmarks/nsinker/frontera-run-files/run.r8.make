
nodes=4
tt=48

for nodes in "1" "2" "4" "8" "16" "32" "64" "128" "256";
do
tasks=$(($nodes * $tt))
echo "nodes=$nodes tpn=$tt tasks=$tasks"
#JJ="-j"

sed "s/&NODES&/$nodes/g" run.r8.base | sed "s/&TASKS&/$tasks/g" > run.r8.$nodes

done
