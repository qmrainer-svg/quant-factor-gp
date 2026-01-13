# 1. 运行完整系统
python gp_stock_mining.py

# 2. 调整参数优化
gp = GeneticProgramming(
    population_size=200,    # 增加种群规模
    max_generations=100,    # 增加进化代数
    max_depth=6,           # 增加最大深度
    crossover_rate=0.85,
    mutation_rate=0.15
)

# 3. 添加真实数据
class RealDataLoader(DataLoader):
    def load_real_data(self):
        # 连接数据库或API获取真实数据
        pass