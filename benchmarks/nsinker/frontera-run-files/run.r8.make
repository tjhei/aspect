
nodes=2048
tt=56
tasks=$(($nodes * $tt))
echo "nodes=$nodes tpn=$tt tasks=$tasks"

aa=100
sed "s/&NODES&/$nodes/g" run.r8.base | sed "s/&TASKS&/$tasks/g" | sed "s/&ADAPTIVE&/$aa/g" > run.r8.$nodes

