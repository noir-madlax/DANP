# 虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行主程序
python main_danp.py

# 克隆巴赫Alpha分析
python ./analysis/cronbach_alpha_analysis.py

# 风险回路分析
python ./analysis/causal_loop_analysis.py

# 因素因果散点图
python ./analysis/dematel_visualization_separate.py