#!bin/sh
echo " nohup sh run.sh > run.log 2>&1 &   8125"
echo "shell批处理脚本  运行日志"
echo "..."
echo "..."
echo "======================================"
python3.7 -u -W ignore run.py --cuda --dataset DBLP --metapath-weight 0.1#0.1#0.8 > log-DBLP1.log 2>&1;
python3.7 -u -W ignore run.py --cuda --dataset DBLP --metapath-weight 0.2#0.2#0.6 > log-DBLP2.log 2>&1;
python3.7 -u -W ignore run.py --cuda --dataset DBLP --metapath-weight 0.3#0.1#0.6 > log-DBLP3.log 2>&1;
python3.7 -u -W ignore run.py --cuda --dataset DBLP --metapath-weight 0.2#0.1#0.7 > log-DBLP4.log 2>&1;
python3.7 -u -W ignore run.py --cuda --dataset DBLP --metapath-weight 0.3#0.3#0.4 > log-DBLP5.log 2>&1;
echo "======================================"
echo "all finished"
