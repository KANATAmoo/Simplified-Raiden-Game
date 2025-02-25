import pygame
import numpy as np
import random
import math
from collections import defaultdict  # 自动处理未见过的新状态

# 初始化Pygame
pygame.init()

# 游戏窗口参数
WIDTH = 800  # 宽度
HEIGHT = 600  # 高度

# Q-learning参数
GRID_SIZE = 15  # 状态离散化的网格尺寸（值越小状态划分越精细）
LEARNING_RATE = 0.05  # 学习率α（0.05表示保留95%旧知识，吸收5%新知识）
DISCOUNT_FACTOR = 0.97  # 折扣因子γ（0.97表示重视远期奖励）
EPSILON_START = 1.0  # 初始探索概率（100%随机探索）
EPSILON_END = 0.01  # 最小探索概率（保留1%的随机探索）
EPSILON_DECAY = 0.995  # 探索概率衰减系数（每轮乘以该值）
EPISODES = 2000  # 总训练轮次

# 动态难度参数（简单→困难）

# 敌机速度从3线性增加到6
INIT_ENEMY_SPEED = 5  # 初始敌机下落速度
MAX_ENEMY_SPEED = 8  # 最大敌机下落速度
# 敌机开火概率从2%增加到5%
INIT_FIRE_RATE = 0.02  # 初始敌机开火概率
MAX_FIRE_RATE = 0.05  # 最大敌机开火概率
# 最大敌机数量从100增加到1000
INIT_MAX_ENEMIES = 200  # 初始最大敌机数量
MAX_ENEMY_COUNT = 1000  # 最终最大敌机数量
DIFFICULTY_INTERVAL = 30  # 难度提升间隔（每30轮提升一次难度）

# 玩家参数
PLAYER_SPEED = 9  # 玩家移动速度
PLAYER_COLOR = (0, 0, 255)  # 颜色（RGB蓝色）

# 子弹参数
BULLET_SPEED = {  # 子弹速度配置字典
    'player': 11,  # 玩家子弹速度（较快）
    'enemy': 9  # 敌人子弹速度（稍慢）
}

# 颜色
WHITE = (255, 255, 255)  # 文字颜色
RED = (255, 0, 0)  # 敌机颜色（红色）
YELLOW = (255, 255, 0)  # 玩家子弹颜色（黄色）
BACKGROUND = (0, 0, 0)  # 背景颜色（黑色）


# 强化学习Agent
class EnhancedAgent:
    def __init__(self):
        # Q表：使用默认字典自动处理未知状态，每个状态有5个动作的Q值
        self.q_table = defaultdict(lambda: np.zeros(5))
        self.prev_state = None  # 存储前一个状态（时序差分学习需要）
        self.prev_action = None  # 存储前一个动作（用于更新Q值）
        self.epsilon = EPSILON_START  # 当前探索率（动态衰减）

    # 将连续状态离散化为机器学习可处理的特征向量
    def discretize_state(self, player_rect, bullets):
        # 玩家网格位置（离散化核心特征）
        # 参数：
        # player_rect: pygame.Rect - 玩家飞机的位置和尺寸
        # bullets: pygame.sprite.Group - 当前所有子弹
        grid_x = min(player_rect.x // GRID_SIZE, (WIDTH // GRID_SIZE) - 1)
        grid_y = min(player_rect.y // GRID_SIZE, (HEIGHT // GRID_SIZE) - 1)

        # 获取最近的三个敌人子弹（按欧氏距离排序）
        enemy_bullets = [b for b in bullets if b.owner == "enemy"]
        sorted_bullets = sorted(
            enemy_bullets,
            key=lambda b: math.hypot(b.rect.x - player_rect.x,
                                     b.rect.y - player_rect.y)
        )[:3]  # 只保留前三个最近的子弹

        # 构建威胁特征向量（方向+距离）
        state_features = []
        for bullet in sorted_bullets:
            dx = bullet.rect.x - player_rect.x  # x轴方向差
            dy = bullet.rect.y - player_rect.y  # y轴方向差
            angle = math.degrees(math.atan2(dy, dx))  # 计算子弹相对角度（-180~180）
            direction = int((angle + 180) // 45) % 8 + 1  # 转换为1-8方向编码
            distance = math.hypot(dx, dy) // 50  # 离散化距离（每50像素为单位）
            state_features.extend([direction, int(distance)])

        # 填充不足三个子弹的情况（保证特征维度一致）
        while len(state_features) < 6:  # 3个子弹×2特征=6
            state_features.extend([0, 0])  # 用0表示无威胁

        return (grid_x, grid_y, *state_features)  # 拼接所有特征

    # 贪心策略选择动作
    def choose_action(self, state):
        if random.random() < self.epsilon:  # 探索：随机选择动作
            return random.randint(0, 4) # 动作编号（0-4对应不动、左、右、上、下）
        else:  # 选择当前Q值最高的动作
            return np.argmax(self.q_table[state])

    # 指数衰减探索率，每轮训练后调用
    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    # 使用Q-learning算法更新Q表
    def update_q_table(self, state, reward):
        if self.prev_state is not None:
            old_q = self.q_table[self.prev_state][self.prev_action]  # 旧Q值
            next_max_q = np.max(self.q_table[state])  # 下一状态最大Q值

            # 动作变化惩罚系数（减少无效抖动）
            action_penalty = 0.1 if self.prev_action != np.argmax(self.q_table[state]) else 0

            # 计算新Q值
            # 更新公式：Q(s,a) = (1-α)Q + α(r - penalty + γmaxQ')
            new_q = (1 - LEARNING_RATE) * old_q + \
                    LEARNING_RATE * (reward - action_penalty + DISCOUNT_FACTOR * next_max_q)

            # 更新Q表
            self.q_table[self.prev_state][self.prev_action] = new_q


# 游戏对象类
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()  # 调用父类初始化
        self.image = pygame.Surface((30, 30))  # 30x30像素
        self.image.fill(PLAYER_COLOR)  # 填充颜色
        self.rect = self.image.get_rect(center=(WIDTH // 2, HEIGHT - 50))  # 初始位置
        self.speed = PLAYER_SPEED  # 设置移动速度

    def update(self, action=None):
        """根据动作更新玩家位置"""
        if action is not None:  # None为不执行动作
            if action == 1:  # 左移
                self.rect.x -= self.speed
            elif action == 2:  # 右移
                self.rect.x += self.speed
            elif action == 3:  # 上移
                self.rect.y -= self.speed
            elif action == 4:  # 下移
                self.rect.y += self.speed

        # 强制边界限制，防止移出屏幕
        self.rect.clamp_ip(pygame.Rect(10, 0, WIDTH - 20, HEIGHT))  # 左右保留10像素边距


class AdaptiveEnemy(pygame.sprite.Sprite):
    def __init__(self, episode):
        super().__init__()
        self.image = pygame.Surface((25, 25))
        self.image.fill(RED)
        self.rect = self.image.get_rect(center=(random.randint(30, WIDTH - 30), 30))

        # 动态难度计算（线性递增）
        progress = min(episode / EPISODES, 1.0)
        self.speed = INIT_ENEMY_SPEED + (MAX_ENEMY_SPEED - INIT_ENEMY_SPEED) * progress
        self.fire_rate = INIT_FIRE_RATE + (MAX_FIRE_RATE - INIT_FIRE_RATE) * progress

    def update(self):
        """自动下落并概率性开火"""
        self.rect.y += self.speed
        if random.random() < self.fire_rate:
            bullets.add(Bullet(self.rect.center, "enemy"))


class Bullet(pygame.sprite.Sprite):
    def __init__(self, pos, owner):
        super().__init__()
        self.image = pygame.Surface((4, 10))
        self.image.fill(YELLOW if owner == "player" else RED)
        self.rect = self.image.get_rect(center=pos)
        self.speed = BULLET_SPEED[owner]
        self.owner = owner

    def update(self):
        """根据所有者更新移动方向"""
        if self.owner == "player":
            self.rect.y -= self.speed
        else:
            self.rect.y += self.speed
        if self.rect.bottom < 0 or self.rect.top > HEIGHT:
            self.kill()

# 初始化游戏窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("雷电")  # 标题
font = pygame.font.Font(None, 24)  # 字体（默认字体，24号）

# 全局游戏状态
best_time = 0  # 历史最佳存活时间（毫秒）
best_episode = 0  # 最佳表现轮次编号
current_difficulty = 0  # 当前难度等级（0为初始难度）


# 碰撞检测函数
def check_collisions():
    # True表示发生碰撞，游戏结束
    # 敌机碰撞
    if pygame.sprite.spritecollideany(player, enemies):
        return True
    # 子弹碰撞
    for bullet in pygame.sprite.spritecollide(player, bullets, False):
        if bullet.owner == "enemy":
            return True
    return False


# 智能奖励函数
def calculate_reward(player_rect, bullets, prev_min_distance):
    enemy_bullets = [b for b in bullets if b.owner == "enemy"]
    reward = 0.5  # 基础存活奖励

    if enemy_bullets:
        # 计算所有子弹的欧氏距离
        distances = [math.hypot(b.rect.x - player_rect.x,
                                b.rect.y - player_rect.y)
                     for b in enemy_bullets]
        min_distance = min(distances)  # 最近子弹距离

        # 距离变化奖励（鼓励远离危险）
        distance_change = (min_distance - prev_min_distance) * 0.1

        # 动态危险系数（距离越近惩罚越大）
        danger_coeff = 1 / (1 + math.sqrt(min_distance))  # 平方根平滑

        # 组合奖励组件
        reward += min_distance * 0.05 + distance_change - danger_coeff * 0.3
    else:
        min_distance = 100  # 无子弹时默认安全距离

    # 边界惩罚（鼓励保持在中央）
    if player_rect.left < 20 or player_rect.right > WIDTH - 20:
        reward -= 0.5

    return reward, min_distance


# 主训练循环
all_sprites = pygame.sprite.Group()  # 管理所有敌机
enemies = pygame.sprite.Group()  # 敌机专用组（用于碰撞检测）
bullets = pygame.sprite.Group()  # 子弹管理组
player = Player()  # 创建玩家实例
agent = EnhancedAgent()  # 创建强化学习智能体

for episode in range(EPISODES):  # 开始训练循环
    # 重置游戏状态
    player.rect.center = (WIDTH // 2, HEIGHT - 50)  # 玩家复位到初始位置
    enemies.empty()  # 清空敌机组
    bullets.empty()  # 清空子弹组
    all_sprites.empty()  # 清空所有精灵

    total_reward = 0  # 本轮累计奖励
    done = False  # 本轮是否结束标志
    frame_count = 0  # 帧计数器（防止无限循环）
    start_time = pygame.time.get_ticks()  # 记录本轮开始时间
    prev_min_distance = 100  # 初始化安全距离

    # 计算当前难度参数
    current_difficulty = min(episode // DIFFICULTY_INTERVAL,
                             EPISODES // DIFFICULTY_INTERVAL)
    max_enemies = INIT_MAX_ENEMIES + int(
        current_difficulty * (MAX_ENEMY_COUNT - INIT_MAX_ENEMIES) /
        (EPISODES // DIFFICULTY_INTERVAL))

    while not done:  # 本轮游戏循环
        screen.fill(BACKGROUND)  # 用黑色清空画面

        # 渲染信息面板
        info_texts = [
            f"Round: {episode + 1}/{EPISODES}",
            f"Best Time: {best_time // 60000:02}:{(best_time % 60000) // 1000:02}",
            f"Best Round: {best_episode}",
            f"Difficulty: {current_difficulty}",
            f"Explore: {agent.epsilon:.2f}"
        ]

        # 动态右对齐渲染信息文本
        for idx, text in enumerate(info_texts):
            text_surface = font.render(text, True, WHITE)  # 创建文字表面
            x_pos = WIDTH - text_surface.get_width() - 10  # 右侧留10像素边距
            screen.blit(text_surface, (x_pos, 10 + idx * 25))  # 绘制文字

        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # 点击关闭按钮
                pygame.quit()  # 关闭Pygame
                quit()  # 退出程序

        # 敌人生成逻辑
        if random.random() < 0.03 and len(enemies) < max_enemies:
            enemy = AdaptiveEnemy(episode)  # 创建自适应敌机
            all_sprites.add(enemy)  # 添加到精灵组
            enemies.add(enemy)  # 添加到敌机组

        # 智能体决策流程
        state = agent.discretize_state(player.rect, bullets)  # 获取当前状态
        action = agent.choose_action(state)  # 选择动作
        player.update(action)  # 执行动作

        # 更新游戏对象状态
        all_sprites.update()  # 更新所有敌机（移动+射击）
        bullets.update()  # 更新所有子弹位置

        # 计算存活时间
        current_time = pygame.time.get_ticks() - start_time  # 计算已存活时间
        minutes = current_time // 60000  # 转换为分钟
        seconds = (current_time % 60000) // 1000  # 转换为秒

        # 碰撞检测
        collision = check_collisions()  # 执行碰撞检测
        if collision:  # 发生碰撞
            reward = -1000  # 大额惩罚
            done = True  # 结束本轮
        else:  # 安全状态
            reward, current_min_distance = calculate_reward(
                player.rect, bullets, prev_min_distance)
            prev_min_distance = current_min_distance  # 更新距离记录

        # Q-learning更新
        if agent.prev_state is not None:  # 跳过第一个状态
            agent.update_q_table(state, reward)
        agent.prev_state = state  # 存储当前状态
        agent.prev_action = action  # 存储当前动作
        total_reward += reward  # 累计总奖励

        # 渲染游戏对象
        all_sprites.draw(screen)  # 绘制所有敌机
        bullets.draw(screen)  # 绘制所有子弹
        screen.blit(player.image, player.rect)  # 绘制玩家

        # 绘制当前存活时间
        time_text = font.render(f"Time: {minutes:02}:{seconds:02}", True, WHITE)
        screen.blit(time_text, (10, 10))  # 左上角显示

        pygame.display.flip()  # 更新整个显示表面
        pygame.time.Clock().tick(30)  # 控制帧率为30FPS

        frame_count += 1  # 帧数累加
        if frame_count > 3000:  # 安全机制：超过3000帧强制结束
            done = True

    # 更新全局记录
    if current_time > best_time:  # 打破历史记录
        best_time = current_time
        best_episode = episode + 1  # 记录最佳轮次（episode从0开始）
        print(f"★新纪录★ 轮次{best_episode} 存活{best_time // 1000}秒")

    # 衰减探索率
    agent.update_epsilon()

    # 打印训练进度
    print(f"轮次: {episode + 1:04d} | 时间: {minutes:02}:{seconds:02} | "
          f"奖励: {total_reward:7.1f} | ε: {agent.epsilon:.3f} | "
          f"难度: {current_difficulty}")

pygame.quit()