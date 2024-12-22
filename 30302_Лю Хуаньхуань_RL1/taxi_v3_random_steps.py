import gym

# Создайте среду Taxi-v3 и укажите режим рендеринга в виде текста (ansi).;创建 Taxi-v3 环境，指定渲染模式为文本（ansi）
env = gym.make("Taxi-v3", render_mode="ansi")

# Инициализировать среду;初始化环境
state = env.reset()
print(f"Initial State: {state}")

# Выполните 10 случайных действий.;执行 10 次随机动作
for step in range(10):
    print(f"\nStep {step + 1}:")
    print(env.render())  # Отобразить текущую среду в виде текста;以文本方式显示当前环境
    action = env.action_space.sample()  # Случайный выбор действий;随机选择动作
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"State: {next_state}, Action: {action}, Reward: {reward}, Done: {terminated or truncated}")
    if terminated or truncated:  # Сбросить среду, если ход заканчивается или усекается;如果回合结束或被截断，则重置环境
        state = env.reset()  # Сбросить среду;

env.close()