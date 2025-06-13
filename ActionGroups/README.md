

# ActionGroups 目录说明

本文件夹包含为 Tonypi 机器人设计的一系列动作组（Action Groups），每个动作组以 `.d6a` 文件形式存储。此外，还包含一个 Python 脚本 `convert.py`，可用于与这些动作组相关的处理操作。

## 文件结构

- `.d6a` 文件：每个文件对应一个动作组，文件名即为动作组功能的简要描述（例如：`dance.d6a` 表示舞蹈动作组）。
- `convert.py`：用于动作组文件的格式转换或批量处理（请参考脚本内注释以了解具体功能）。

## 动作组文件列表及简要说明

| 文件名                           | 说明                       |
| -------------------------------- | -------------------------- |
| catch_ball_left_move_up.d6a      | 向左接球并上移动作         |
| catch_ball_right_move.d6a        | 向右接球动作               |
| catch_ball_right_move_up.d6a     | 向右接球并上移动作         |
| catch_ball_turn_left.d6a         | 接球并左转动作             |
| catch_ball_turn_left_up.d6a      | 接球左转并上移动作         |
| catch_ball_turn_right.d6a        | 接球并右转动作             |
| catch_ball_turn_right_up.d6a     | 接球右转并上移动作         |
| catch_ball_up.d6a                | 接球并上移动作             |
| chest.d6a                        | 胸部相关动作               |
| climb_stairs.d6a                 | 爬楼梯动作                 |
| climb_stairs_1.d6a               | 爬楼梯动作（变体）         |
| creep_forward.d6a                | 匍匐前进动作               |
| dance.d6a                        | 跳舞动作                   |
| down_floor.d6a                   | 下楼动作                   |
| down_floor_1.d6a                 | 下楼动作（变体）           |
| down_objec.d6a                   | 下移物体动作               |
| gb.d6a / gb2.d6a                 | 其他动作组（具体待补充）   |
| go.d6a                           | 前进动作                   |
| go_forward.d6a                   | 向前行动作                 |
| go_forward_end.d6a               | 前进结束动作               |
| go_forward_fast.d6a              | 快速前进动作               |
| go_forward_one_small_step.d6a    | 微小步前进动作             |
| go_forward_one_step.d6a          | 单步前进动作               |

> **注意**：如需详细了解每个动作组的内容，请结合具体硬件平台或查看 `convert.py` 脚本的使用方法。

## 使用方法

1. 将所需的 `.d6a` 文件上传至 Tonypi 机器人或相关控制平台。
2. 通过机器人控制系统调用对应的动作组文件，即可执行相应动作。
3. 如需批量处理或转换动作组文件，可使用 `convert.py` 脚本（具体用法请参考脚本内部说明）。

## 贡献与补充

如需新增或补充动作组说明，请按照表格格式添加，并确保文件命名规范、描述清晰。

