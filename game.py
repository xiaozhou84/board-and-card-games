import tkinter as tk #tkinter模块
from tkinter import messagebox, filedialog #tkinter模块
from abc import ABC, abstractmethod #抽象基类
from enum import Enum #枚举类
import json #json模块
from typing import List, Tuple, Optional, Set, Dict, Any #类型注解

class StoneColor(Enum):#棋子颜色枚举
    BLACK = 1
    WHITE = 2
    EMPTY = 0

class GameType(Enum):#游戏类型枚举
    GO = "围棋"
    GOMOKU = "五子棋"

class GameError(Exception):#游戏相关异常基类
    """游戏相关异常基类"""
    pass

class InvalidMoveError(GameError):#无效落子异常
    """无效落子异常"""
    pass

class InvalidBoardSizeError(GameError):#无效棋盘大小异常
    """无效棋盘大小异常"""
    pass

class SaveLoadError(GameError):#保存/加载游戏异常
    """保存/加载游戏异常"""
    pass

class GameMemento:#游戏备忘录类，用于保存游戏状态
    """游戏备忘录类，用于保存游戏状态"""
    def __init__(self, board: List[List[StoneColor]], current_player: StoneColor, #游戏备忘录类初始化
                 last_move: Optional[Tuple[int, int]], move_history: List[Tuple[int, int]],
                 game_over: bool, winner: Optional[StoneColor], resigned: bool,
                 pass_count: int, undo_count: int):
        self.board = [[stone for stone in row] for row in board]  # 深拷贝棋盘
        self.current_player = current_player#当前玩家
        self.last_move = last_move#最后落子位置
        self.move_history = move_history.copy()  # 深拷贝移动历史
        self.game_over = game_over#游戏是否结束
        self.winner = winner#胜利者
        self.resigned = resigned#认输
        self.pass_count = pass_count#跳过回合次数
        self.undo_count = undo_count#悔棋次数

class GameState:#游戏状态类，负责维护游戏状态
    """游戏状态类，负责维护游戏状态"""
    def __init__(self, board_size: int, game_type: GameType):#游戏状态类初始化
        if not (8 <= board_size <= 19):
            raise InvalidBoardSizeError("棋盘大小必须在8到19之间")
        self.board_size = board_size  # 棋盘大小
        self.game_type = game_type  # 游戏类型
        self.board = [[StoneColor.EMPTY for _ in range(board_size)] for _ in range(board_size)]  # 棋盘
        self.current_player = StoneColor.BLACK  # 当前玩家
        self.last_move = None  # 最后落子位置
        self.move_history = []  # 移动历史
        self.game_over = False  # 游戏是否结束
        self.winner = None  # 胜利者
        self.resigned = False  # 认输
        self.pass_count = 0  # 跳过回合次数
        self.undo_count = 0  # 悔棋次数
        self.max_undo_steps = 3  # 最大悔棋步数
        self._mementos = []  # 存储备忘录的列表

    def create_memento(self) -> GameMemento:#游戏状态类创建备忘录
        """创建当前状态的备忘录"""
        return GameMemento(
            self.board,#棋盘
            self.current_player,#当前玩家
            self.last_move,#最后落子位置
            self.move_history,#移动历史
            self.game_over,#游戏是否结束
            self.winner,#胜利者
            self.resigned,#认输
            self.pass_count,#跳过回合次数
            self.undo_count#悔棋次数
        )

    def restore_from_memento(self, memento: GameMemento) -> None:#游戏状态类从备忘录恢复状态
        """从备忘录恢复状态"""
        self.board = [[stone for stone in row] for row in memento.board]#棋盘
        self.current_player = memento.current_player#当前玩家
        self.last_move = memento.last_move#最后落子位置
        self.move_history = memento.move_history.copy()#移动历史
        self.game_over = memento.game_over#游戏是否结束
        self.winner = memento.winner#胜利者
        self.resigned = memento.resigned#认输
        self.pass_count = memento.pass_count#跳过回合次数
        self.undo_count = memento.undo_count#悔棋次数

    def save_state(self) -> None:#游戏状态类保存状态到备忘录列表
        """保存当前状态到备忘录列表"""
        if len(self._mementos) >= self.max_undo_steps:#如果备忘录列表长度大于等于最大悔棋步数
            self._mementos.pop(0)  # 移除最旧的备忘录
        self._mementos.append(self.create_memento())#添加当前状态到备忘录列表

    def restore_previous_state(self) -> bool:#游戏状态类恢复到上一个状态
        """恢复到上一个状态"""
        if not self._mementos:#如果备忘录列表为空
            return False
        memento = self._mementos.pop()#移除最后一个备忘录
        self.restore_from_memento(memento)#恢复到上一个状态
        return True

    def is_valid_position(self, x: int, y: int) -> bool:#游戏状态类判断落子位置是否有效
        return 0 <= x < self.board_size and 0 <= y < self.board_size#判断落子位置是否在棋盘范围内

    def get_adjacent_positions(self, x: int, y: int) -> List[Tuple[int, int]]:#游戏状态类获取相邻位置
        positions = []#相邻位置列表
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:#四个方向
            new_x, new_y = x + dx, y + dy#新位置
            if self.is_valid_position(new_x, new_y):#判断新位置是否在棋盘范围内
                positions.append((new_x, new_y))#添加新位置到相邻位置列表
        return positions#返回相邻位置列表

class GameEvent:#游戏事件基类
    """游戏事件基类"""
    def __init__(self, event_type: str, data: Any = None):#游戏事件基类初始化
        self.event_type = event_type#事件类型
        self.data = data#事件数据

class GameObserver(ABC):#游戏观察者接口
    """游戏观察者接口"""
    @abstractmethod
    def on_game_event(self, event: GameEvent) -> None:#游戏观察者接口
        pass

class GameSubject:#游戏主题类，负责管理观察者
    """游戏主题类，负责管理观察者"""
    def __init__(self):#游戏主题类初始化
        self._observers: List[GameObserver] = []

    def attach(self, observer: GameObserver) -> None:#游戏主题类附加观察者
        if observer not in self._observers:#如果观察者不存在于观察者列表中
            self._observers.append(observer)#添加观察者

    def detach(self, observer: GameObserver) -> None:#游戏主题类分离观察者
        if observer in self._observers:#如果观察者存在于观察者列表中
            self._observers.remove(observer)#移除观察者

    def notify(self, event: GameEvent) -> None:#游戏主题类通知观察者
        for observer in self._observers:#遍历观察者列表
            observer.on_game_event(event)#通知观察者

class PlatformBackend(ABC):#平台后端抽象类
    """平台后端抽象类"""
    def __init__(self):#平台后端抽象类初始化
        self.state = None#游戏状态
        self._observers = []#观察者列表
        self._mementos = []#备忘录列表

    def attach(self, observer):#平台后端抽象类附加观察者
        if observer not in self._observers:#如果观察者不存在于观察者列表中
            self._observers.append(observer)#添加观察者

    def detach(self, observer):#平台后端抽象类分离观察者
        if observer in self._observers:#如果观察者存在于观察者列表中
            self._observers.remove(observer)#移除观察者

    def notify(self, event):#平台后端抽象类通知观察者
        for observer in self._observers:#遍历观察者列表
            observer.on_game_event(event)#通知观察者

    def save_state(self):#平台后端抽象类保存状态到备忘录
        """保存当前状态到备忘录"""
        if len(self._mementos) >= 3:#如果备忘录列表长度大于等于3
            self._mementos.pop(0)#移除最旧的备忘录
        self._mementos.append(self.create_memento())#添加当前状态到备忘录列表

    def restore_state(self):#平台后端抽象类恢复状态
        """从备忘录恢复状态"""
        if not self._mementos:#如果备忘录列表为空
            return False
        memento = self._mementos.pop()#移除最后一个备忘录
        self.restore_from_memento(memento)#恢复到上一个状态
        return True

    @abstractmethod
    def create_memento(self):#平台后端抽象类创建备忘录
        pass

    @abstractmethod
    def restore_from_memento(self, memento):#平台后端抽象类从备忘录恢复状态
        pass

    @abstractmethod
    def make_move(self, x: int, y: int) -> bool:#平台后端抽象类落子
        pass

    @abstractmethod
    def undo(self) -> bool:#平台后端抽象类悔棋
        pass

    @abstractmethod
    def pass_turn(self) -> None:#平台后端抽象类跳过回合
        pass

    @abstractmethod
    def resign(self) -> None:#平台后端抽象类认输
        pass

    @abstractmethod
    def get_game_state(self):#平台后端抽象类获取游戏状态
        pass

class GoBackend(PlatformBackend):#围棋后端实现
    """围棋后端实现"""
    def __init__(self, board_size: int):#围棋后端实现初始化
        super().__init__()#调用父类初始化
        self.state = GameState(board_size, GameType.GO)#游戏状态
        self.board_size = board_size#棋盘大小
        self.last_captured_stones = []  # 记录上一次提子
        self.last_liberties = {}  # 记录上一次的气

    def calculate_territory(self) -> Tuple[float, float]:
        """计算黑白双方的领地"""
        black_territory = 0
        white_territory = 0
        visited = set()

        def flood_fill(x: int, y: int) -> Tuple[Set[Tuple[int, int]], Set[StoneColor]]:
            """使用洪水填充算法计算一个区域的边界"""
            if (x, y) in visited:
                return set(), set()
            
            visited.add((x, y))
            area = {(x, y)}
            boundaries = set()
            
            for nx, ny in self.state.get_adjacent_positions(x, y):
                if (nx, ny) not in visited:
                    if self.state.board[nx][ny] == StoneColor.EMPTY:
                        area.add((nx, ny))
                        visited.add((nx, ny))
                    else:
                        boundaries.add(self.state.board[nx][ny])
            
            return area, boundaries

        # 遍历所有空点
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.state.board[i][j] == StoneColor.EMPTY and (i, j) not in visited:
                    area, boundaries = flood_fill(i, j)
                    if len(boundaries) == 1:  # 如果区域只被一种颜色包围
                        color = boundaries.pop()
                        if color == StoneColor.BLACK:
                            black_territory += len(area)
                        else:
                            white_territory += len(area)

        # 加上各自的活子数
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.state.board[i][j] == StoneColor.BLACK:
                    black_territory += 1
                elif self.state.board[i][j] == StoneColor.WHITE:
                    white_territory += 1

        return black_territory, white_territory

    def check_win_condition(self) -> Optional[StoneColor]:
        """检查胜负条件"""
        if self.state.pass_count >= 2:  # 双方都选择跳过
            black_territory, white_territory = self.calculate_territory()
            if black_territory > white_territory:
                return StoneColor.BLACK
            elif white_territory > black_territory:
                return StoneColor.WHITE
            else:
                return None  # 平局
        return None

    def create_memento(self):#围棋后端实现创建备忘录
        return GameMemento(
            self.state.board,#棋盘
            self.state.current_player,#当前玩家
            self.state.last_move,#最后落子位置
            self.state.move_history,#移动历史
            self.state.game_over,#游戏是否结束
            self.state.winner,#胜利者
            self.state.resigned,#认输
            self.state.pass_count,#跳过回合次数
            self.state.undo_count#悔棋次数
        )

    def restore_from_memento(self, memento):#围棋后端实现从备忘录恢复状态
        self.state.board = [[stone for stone in row] for row in memento.board]#棋盘
        self.state.current_player = memento.current_player#当前玩家
        self.state.last_move = memento.last_move#最后落子位置
        self.state.move_history = memento.move_history.copy()#移动历史
        self.state.game_over = memento.game_over#游戏是否结束
        self.state.winner = memento.winner#胜利者
        self.state.resigned = memento.resigned#认输
        self.state.pass_count = memento.pass_count#跳过回合次数
        self.state.undo_count = memento.undo_count#悔棋次数
        # 恢复提子和气
        self.last_captured_stones = []
        self.last_liberties = {}
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.state.board[i][j] != StoneColor.EMPTY:
                    self.last_liberties[(i, j)] = self.get_group_liberties(i, j)

    def make_move(self, x: int, y: int) -> bool:#围棋后端实现落子
        if self.state.game_over:#如果游戏已结束
            raise InvalidMoveError("游戏已结束")#抛出无效落子异常
        
        if not self.is_valid_move(x, y):#如果落子位置无效
            return False#返回False
        
        self.save_state()#保存状态到备忘录列表
        
        # 放置棋子
        self.state.board[x][y] = self.state.current_player#放置棋子
        self.state.last_move = (x, y)#最后落子位置
        self.state.move_history.append((x, y))#移动历史
        self.state.pass_count = 0#跳过回合次数
        
        # 移除死子
        self.remove_dead_stones(x, y)#移除死子
        
        # 切换玩家
        self.state.current_player = StoneColor.WHITE if self.state.current_player == StoneColor.BLACK else StoneColor.BLACK#切换玩家
        
        # 通知观察者
        self.notify(GameEvent("BOARD_UPDATE", {
            "board_size": self.board_size,
            "game_type": GameType.GO
        }))
        
        return True

    def undo(self) -> bool:#围棋后端实现悔棋
        if self.state.game_over:#如果游戏已结束
            raise InvalidMoveError("游戏已结束，不能悔棋")#抛出无效落子异常
        if not self._mementos:#如果备忘录列表为空
            raise InvalidMoveError("没有可悔的棋")#抛出无效落子异常
        if self.state.undo_count >= 3:#如果悔棋次数大于等于3
            raise InvalidMoveError("悔棋次数超过限制（最多3次）")#抛出无效落子异常
        
        if self.restore_state():#恢复状态
            self.state.undo_count += 1#悔棋次数加1
            # 恢复提子
            for x, y in self.last_captured_stones:
                self.state.board[x][y] = StoneColor.WHITE if self.state.current_player == StoneColor.BLACK else StoneColor.BLACK
            # 重新计算气
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.state.board[i][j] != StoneColor.EMPTY:
                        self.last_liberties[(i, j)] = self.get_group_liberties(i, j)
            self.notify(GameEvent("BOARD_UPDATE", {
                "board_size": self.board_size,
                "game_type": GameType.GO
            }))#通知观察者
            return True
        return False

    def pass_turn(self) -> None:#围棋后端实现跳过回合
        if self.state.game_over:#如果游戏已结束
            raise InvalidMoveError("游戏已结束")#抛出无效落子异常
        self.state.current_player = StoneColor.WHITE if self.state.current_player == StoneColor.BLACK else StoneColor.BLACK#切换玩家
        self.state.pass_count += 1#跳过回合次数加1
        if self.state.pass_count >= 2:#如果跳过回合次数大于等于2
            winner = self.check_win_condition()
            self.state.game_over = True#游戏结束
            self.state.winner = winner
            self.notify(GameEvent("GAME_OVER", winner))#通知观察者

    def resign(self) -> None:#围棋后端实现认输
        if self.state.game_over:#如果游戏已结束
            raise InvalidMoveError("游戏已结束")#抛出无效落子异常
        self.state.game_over = True#游戏结束
        self.state.winner = StoneColor.WHITE if self.state.current_player == StoneColor.BLACK else StoneColor.BLACK#胜利者
        self.state.resigned = True#认输
        self.notify(GameEvent("GAME_OVER", self.state.winner))#通知观察者

    def get_game_state(self):#围棋后端实现获取游戏状态
        return self.state

    def is_valid_move(self, x: int, y: int) -> bool:#围棋后端实现判断落子位置是否有效
        if not self.state.is_valid_position(x, y):#如果落子位置无效
            raise InvalidMoveError("落子位置超出棋盘范围")#抛出无效落子异常
        
        if self.state.board[x][y] != StoneColor.EMPTY:#如果落子位置已有棋子
            raise InvalidMoveError("该位置已有棋子")#抛出无效落子异常
        
        # 临时放置棋子检查气
        self.state.board[x][y] = self.state.current_player#放置棋子
        liberties = self.get_group_liberties(x, y)#获取气
        self.state.board[x][y] = StoneColor.EMPTY#移除棋子
        
        if not liberties:#如果气为0
            raise InvalidMoveError("自杀着法无效")#抛出无效落子异常
        
        return True

    def get_group_liberties(self, x: int, y: int, visited: Set[Tuple[int, int]] = None) -> Set[Tuple[int, int]]:#围棋后端实现获取气
        if visited is None:#如果访问列表为空
            visited = set()#初始化访问列表
        
        if (x, y) in visited:#如果已经访问过
            return set()#返回空集合
        
        visited.add((x, y))#添加到访问列表
        color = self.state.board[x][y]#获取颜色
        liberties = set()#初始化气集合
        
        for adj_x, adj_y in self.state.get_adjacent_positions(x, y):#获取相邻位置
            if self.state.board[adj_x][adj_y] == StoneColor.EMPTY:#如果相邻位置为空
                liberties.add((adj_x, adj_y))#添加到气集合
            elif self.state.board[adj_x][adj_y] == color:#如果相邻位置颜色相同
                liberties.update(self.get_group_liberties(adj_x, adj_y, visited))#递归获取气
        
        return liberties

    def remove_dead_stones(self, x: int, y: int) -> None:#围棋后端实现移除死子
        color = self.state.board[x][y]
        opponent_color = StoneColor.WHITE if color == StoneColor.BLACK else StoneColor.BLACK
        
        # 记录提子
        self.last_captured_stones = []
        
        for i in range(self.board_size):#遍历棋盘
            for j in range(self.board_size):#遍历棋盘
                if self.state.board[i][j] == opponent_color:#如果棋子颜色相同
                    liberties = self.get_group_liberties(i, j)#获取气
                    if not liberties:#如果气为0
                        self.state.board[i][j] = StoneColor.EMPTY#移除棋子
                        self.last_captured_stones.append((i, j))

class GomokuBackend(PlatformBackend):#五子棋后端实现
    """五子棋后端实现"""
    def __init__(self, board_size: int):#五子棋后端实现初始化
        super().__init__()
        self.state = GameState(board_size, GameType.GOMOKU)
        self.board_size = board_size

    def create_memento(self):#五子棋后端实现创建备忘录
        return GameMemento(
            self.state.board, #棋盘
            self.state.current_player, #当前玩家
            self.state.last_move, #最后落子位置
            self.state.move_history, #移动历史
            self.state.game_over, #游戏是否结束
            self.state.winner, #胜利者
            self.state.resigned, #认输
            self.state.pass_count, #跳过回合次数
            self.state.undo_count #悔棋次数
        )

    def restore_from_memento(self, memento):#五子棋后端实现从备忘录恢复状态
        self.state.board = [[stone for stone in row] for row in memento.board]#棋盘
        self.state.current_player = memento.current_player#当前玩家
        self.state.last_move = memento.last_move#最后落子位置
        self.state.move_history = memento.move_history.copy()#移动历史
        self.state.game_over = memento.game_over#游戏是否结束
        self.state.winner = memento.winner#胜利者
        self.state.resigned = memento.resigned#认输
        self.state.pass_count = memento.pass_count#跳过回合次数
        self.state.undo_count = memento.undo_count#悔棋次数

    def make_move(self, x: int, y: int) -> bool:#五子棋后端实现落子
        if self.state.game_over:#如果游戏已结束
            raise InvalidMoveError("游戏已结束")#抛出无效落子异常
        
        if not self.is_valid_move(x, y):#如果落子位置无效
            return False#返回False
        
        self.save_state()
        
        # 放置棋子
        self.state.board[x][y] = self.state.current_player#放置棋子
        self.state.last_move = (x, y)#最后落子位置
        self.state.move_history.append((x, y))#移动历史
        
        # 检查胜利条件
        if self.check_win_condition():#如果胜利条件成立
            self.state.game_over = True#游戏结束
            self.state.winner = self.state.current_player#胜利者
            # 先通知刷新棋盘
            self.notify(GameEvent("BOARD_UPDATE", {
                "board_size": self.board_size,
                "game_type": GameType.GOMOKU
            }))
            # 再通知游戏结束
            self.notify(GameEvent("GAME_OVER", self.state.winner))#游戏结束
            return True
        else:
            # 检查是否平局
            if all(cell != StoneColor.EMPTY for row in self.state.board for cell in row):#如果棋盘所有位置都为空
                self.state.game_over = True#游戏结束
                self.notify(GameEvent("BOARD_UPDATE", {
                    "board_size": self.board_size,
                    "game_type": GameType.GOMOKU
                }))
                self.notify(GameEvent("GAME_OVER", None))
                return True
        # 切换玩家
        self.state.current_player = StoneColor.WHITE if self.state.current_player == StoneColor.BLACK else StoneColor.BLACK
        # 通知观察者
        self.notify(GameEvent("BOARD_UPDATE", {
            "board_size": self.board_size,
            "game_type": GameType.GOMOKU
        }))
        return True

    def undo(self) -> bool:#五子棋后端实现悔棋
        if self.state.game_over:#如果游戏已结束
            raise InvalidMoveError("游戏已结束，不能悔棋")#抛出无效落子异常
        if not self._mementos:#如果备忘录列表为空
            raise InvalidMoveError("没有可悔的棋")#抛出无效落子异常
        if self.state.undo_count >= 3:#如果悔棋次数大于等于3
            raise InvalidMoveError("悔棋次数超过限制（最多3次）")#抛出无效落子异常
        
        if self.restore_state():#恢复状态
            self.state.undo_count += 1#悔棋次数加1
            self.notify(GameEvent("BOARD_UPDATE", {
                "board_size": self.board_size,
                "game_type": GameType.GOMOKU
            }))#通知观察者
            return True
        return False

    def pass_turn(self) -> None:#五子棋后端实现跳过回合
        if self.state.game_over:#如果游戏已结束
            raise InvalidMoveError("游戏已结束")#抛出无效落子异常
        self.state.current_player = StoneColor.WHITE if self.state.current_player == StoneColor.BLACK else StoneColor.BLACK#切换玩家
        self.state.pass_count += 1#跳过回合次数加1
        if self.state.pass_count >= 2:#如果跳过回合次数大于等于2
            self.state.game_over = True#游戏结束
            self.notify(GameEvent("GAME_OVER", None))#通知观察者

    def resign(self) -> None:#五子棋后端实现认输
        if self.state.game_over:#如果游戏已结束
            raise InvalidMoveError("游戏已结束")#抛出无效落子异常
        self.state.game_over = True#游戏结束
        self.state.winner = StoneColor.WHITE if self.state.current_player == StoneColor.BLACK else StoneColor.BLACK#胜利者
        self.state.resigned = True#认输
        self.notify(GameEvent("GAME_OVER", self.state.winner))#通知观察者

    def get_game_state(self):#五子棋后端实现获取游戏状态
        return self.state

    def is_valid_move(self, x: int, y: int) -> bool:#五子棋后端实现判断落子位置是否有效
        if not self.state.is_valid_position(x, y):#如果落子位置无效
            raise InvalidMoveError("落子位置超出棋盘范围")#抛出无效落子异常
        
        if self.state.board[x][y] != StoneColor.EMPTY:#如果落子位置已有棋子
            raise InvalidMoveError("该位置已有棋子")#抛出无效落子异常
        
        return True

    def check_win_condition(self) -> bool:#五子棋后端实现检查胜利条件
        if not self.state.last_move:#如果最后落子位置为空
            return False#返回False
        
        x, y = self.state.last_move#最后落子位置
        color = self.state.board[x][y]#落子颜色
        
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]#方向
        
        for dx, dy in directions:#遍历方向
            count = 1#计数
            for direction in [1, -1]:#遍历方向
                step = 1#步长
                while True:#循环
                    new_x = x + dx * step * direction#新位置
                    new_y = y + dy * step * direction#新位置
                    if not self.state.is_valid_position(new_x, new_y):#如果新位置无效
                        break#如果新位置无效
                    if self.state.board[new_x][new_y] != color:#如果新位置颜色不同
                        break#跳出循环
                    count += 1#计数加1
                    step += 1#步长加1
            
            if count >= 5:#如果计数大于等于5
                return True#返回True
        
        return False#返回False

class GameClient(ABC):#游戏客户端抽象类
    """游戏客户端抽象类"""
    def __init__(self, backend: PlatformBackend):#游戏客户端抽象类初始化
        self.backend = backend#后端

    @abstractmethod
    def handle_click(self, x: int, y: int) -> None:#游戏客户端抽象类处理点击
        pass

    @abstractmethod
    def handle_undo(self) -> None:#游戏客户端抽象类处理悔棋
        pass

    @abstractmethod
    def handle_pass(self) -> None:#游戏客户端抽象类处理跳过回合
        pass

    @abstractmethod
    def handle_resign(self) -> None:#游戏客户端抽象类处理认输
        pass

    @abstractmethod
    def update_display(self) -> None:#游戏客户端抽象类更新显示
        pass

    @abstractmethod
    def on_game_event(self, event: GameEvent) -> None:#游戏客户端抽象类处理游戏事件
        pass

class GUIClient(GameClient):#GUI客户端实现
    """GUI客户端实现"""
    def __init__(self, backend: PlatformBackend, root: tk.Tk):#GUI客户端实现初始化
        super().__init__(backend)#游戏客户端抽象类初始化
        self.root = root#根窗口
        self.setup_ui()#设置UI
        self.update_display()  # 初始化时更新显示

    def setup_ui(self):#GUI客户端实现设置UI
        self.frame = tk.Frame(self.root)#创建框架
        self.frame.pack(expand=True, fill='both')#设置框架大小
        
        # 创建控制面板
        self.control_frame = tk.Frame(self.frame)#创建控制面板
        self.control_frame.pack(side=tk.TOP, pady=10)#设置控制面板位置
        
        # 添加状态标签
        self.status_label = tk.Label(self.control_frame, text="", font=("宋体", 14))
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # 创建按钮
        self.undo_btn = tk.Button(self.control_frame, text="悔棋", command=self.handle_undo,
                                font=("宋体", 14), width=10, height=1)
        self.pass_btn = tk.Button(self.control_frame, text="暂停", command=self.handle_pass,
                                font=("宋体", 14), width=10, height=1)
        self.resign_btn = tk.Button(self.control_frame, text="认输", command=self.handle_resign,
                                  font=("宋体", 14), width=10, height=1)
        
        # 放置按钮
        self.undo_btn.pack(side=tk.LEFT, padx=10)
        self.pass_btn.pack(side=tk.LEFT, padx=10)
        self.resign_btn.pack(side=tk.LEFT, padx=10)
        
        # 创建棋盘画布
        self.canvas = tk.Canvas(self.frame, bg='#DEB887')#创建棋盘画布
        self.canvas.pack(expand=True, fill='both')#设置棋盘画布大小
        
        # 设置棋盘参数
        self.stone_size = 30#棋子大小
        self.board_padding = 40#棋盘边距
        
        # 绑定鼠标点击事件
        self.canvas.bind("<Button-1>", self._on_canvas_click)#绑定鼠标点击事件

    def _on_canvas_click(self, event):#GUI客户端实现处理点击
        # 计算点击位置相对于棋盘左上角的坐标
        x = event.x - self.board_padding#计算点击位置相对于棋盘左上角的坐标
        y = event.y - self.board_padding#计算点击位置相对于棋盘左上角的坐标
        
        # 转换为网格坐标
        grid_x = round(x / self.stone_size)#转换为网格坐标
        grid_y = round(y / self.stone_size)#转换为网格坐标
        
        # 检查点击是否在有效范围内
        if (0 <= grid_x < self.backend.board_size and 
            0 <= grid_y < self.backend.board_size):#检查点击是否在有效范围内
            # 检查点击是否在交叉点附近
            if (abs(x - grid_x * self.stone_size) < self.stone_size * 0.3 and 
                abs(y - grid_y * self.stone_size) < self.stone_size * 0.3):#检查点击是否在交叉点附近
                self.handle_click(grid_y, grid_x)#处理点击

    def handle_click(self, x: int, y: int) -> None:#GUI客户端实现处理点击
        try:
            if self.backend.make_move(x, y):#落子
                self.update_display()#更新显示
        except InvalidMoveError as e:
            messagebox.showerror("无效落子", str(e))#显示错误信息

    def handle_undo(self) -> None:#GUI客户端实现处理悔棋
        try:
            if self.backend.undo():#悔棋
                self.update_display()#更新显示
        except InvalidMoveError as e:
            messagebox.showerror("悔棋失败", str(e))#显示错误信息

    def handle_pass(self) -> None:#GUI客户端实现处理跳过回合
        try:
            self.backend.pass_turn()#跳过回合
            self.update_display()#更新显示
        except InvalidMoveError as e:
            messagebox.showerror("暂停失败", str(e))#显示错误信息

    def handle_resign(self) -> None:#GUI客户端实现处理认输
        try:
            self.backend.resign()#认输
            self.update_display()#更新显示
        except InvalidMoveError as e:
            messagebox.showerror("认输失败", str(e))#显示错误信息

    def update_display(self) -> None:#GUI客户端实现更新显示
        self.canvas.delete("all")#删除所有画布
        state = self.backend.get_game_state()#获取游戏状态
        
        # 获取游戏类型
        game_type = "围棋" if isinstance(self.backend, GoBackend) else "五子棋"
        
        # 更新状态标签和按钮状态
        if state.game_over:
            if state.winner:
                winner_text = "黑方" if state.winner == StoneColor.BLACK else "白方"
                self.status_label.config(text=f"{game_type} - 游戏结束 - {winner_text}获胜")
            else:
                self.status_label.config(text=f"{game_type} - 游戏结束 - 平局")
            # 禁用所有按钮
            self.undo_btn.config(state=tk.DISABLED)
            self.pass_btn.config(state=tk.DISABLED)
            self.resign_btn.config(state=tk.DISABLED)
        else:
            current_player = "黑方" if state.current_player == StoneColor.BLACK else "白方"
            self.status_label.config(text=f"{game_type} - 当前回合: {current_player}")
            # 启用所有按钮
            self.undo_btn.config(state=tk.NORMAL)
            self.pass_btn.config(state=tk.NORMAL)
            self.resign_btn.config(state=tk.NORMAL)
        
        # 计算棋盘大小
        canvas_width = self.backend.board_size * self.stone_size + 2 * self.board_padding#计算棋盘大小
        canvas_height = self.backend.board_size * self.stone_size + 2 * self.board_padding#计算棋盘大小
        
        # 设置画布大小
        self.canvas.config(width=canvas_width, height=canvas_height)#设置画布大小
        
        # 绘制网格线，确保棋盘四周完整
        start = self.board_padding#计算网格线起始点
        end = self.board_padding + (self.backend.board_size - 1) * self.stone_size#计算网格线结束点
        for i in range(self.backend.board_size):#遍历棋盘大小
            pos = self.board_padding + i * self.stone_size#计算网格线位置
            # 水平线
            self.canvas.create_line(start, pos, end, pos)#绘制水平线
            # 垂直线
            self.canvas.create_line(pos, start, pos, end)#绘制垂直线
        # 绘制最外侧的边框线，确保四周封闭
        self.canvas.create_rectangle(start, start, end, end)#绘制最外侧的边框线，确保四周封闭
        
        # 绘制坐标标签
        label_font = ("宋体", 10)#设置标签字体
        for i in range(self.backend.board_size):#遍历棋盘大小
            # 列标签（字母）
            x = self.board_padding + i * self.stone_size#计算列标签位置
            y = self.board_padding - 20#计算列标签位置
            label = chr(65 + i)  # A, B, C, ...
            self.canvas.create_text(x, y, text=label, font=label_font)
            
            # 行标签（数字）
            x = self.board_padding - 20#计算行标签位置
            y = self.board_padding + i * self.stone_size#计算行标签位置
            label = str(self.backend.board_size - i)  # 从下往上编号
            self.canvas.create_text(x, y, text=label, font=label_font)
        
        # 绘制星位点（围棋）
        if isinstance(self.backend, GoBackend):#GUI客户端实现绘制星位点
            star_points = []#初始化星位点
            if self.backend.board_size == 19:#GUI客户端实现绘制星位点
                star_points = [3, 9, 15]#围棋星位点
            elif self.backend.board_size == 13:#GUI客户端实现绘制星位点
                star_points = [3, 6, 9]#围棋星位点
            elif self.backend.board_size == 9:#GUI客户端实现绘制星位点
                star_points = [2, 4, 6]#围棋星位点
            
            for i in star_points:#GUI客户端实现绘制星位点
                for j in star_points:#GUI客户端实现绘制星位点
                    x = self.board_padding + i * self.stone_size#计算星位点位置
                    y = self.board_padding + j * self.stone_size#计算星位点位置
                    self.canvas.create_oval(
                        x - 3, y - 3,
                        x + 3, y + 3,
                        fill='black'
                    )#绘制星位点
        
        # 绘制棋子
        for i in range(self.backend.board_size):#GUI客户端实现绘制棋子
            for j in range(self.backend.board_size):#GUI客户端实现绘制棋子
                stone = state.board[i][j]#获取棋子颜色
                if stone != StoneColor.EMPTY:#GUI客户端实现绘制棋子
                    x = self.board_padding + j * self.stone_size#计算棋子位置
                    y = self.board_padding + i * self.stone_size#计算棋子位置
                    color = "black" if stone == StoneColor.BLACK else "white"#计算棋子颜色
                    
                    # 绘制棋子阴影
                    self.canvas.create_oval(
                        x - self.stone_size//2 + 2,
                        y - self.stone_size//2 + 2,
                        x + self.stone_size//2 + 2,
                        y + self.stone_size//2 + 2,
                        fill='gray'
                    )
                    # 绘制棋子
                    self.canvas.create_oval(
                        x - self.stone_size//2,
                        y - self.stone_size//2,
                        x + self.stone_size//2,
                        y + self.stone_size//2,
                        fill=color
                    )

    def on_game_event(self, event: GameEvent) -> None:#游戏观察者接口
        """处理游戏事件"""
        if event.event_type == "BOARD_UPDATE":#GUI客户端实现处理游戏事件
            self.update_display()
        elif event.event_type == "GAME_OVER":
            if event.data:#GUI客户端实现处理游戏事件
                winner = "黑方" if event.data == StoneColor.BLACK else "白方"#计算胜利者
                messagebox.showinfo("游戏结束", f"{winner}获胜！")#显示胜利者
            else:
                messagebox.showinfo("游戏结束", "平局！")#显示平局

class GameMenu:#游戏菜单组件
    """游戏菜单组件"""
    def __init__(self, root, on_new_game, on_save_game, on_load_game):#游戏菜单组件初始化
        self.menubar = tk.Menu(root)
        
        # 游戏菜单
        game_menu = tk.Menu(self.menubar, tearoff=0)#创建游戏菜单
        game_menu.add_command(label="新建游戏", command=on_new_game)#添加新建游戏命令
        game_menu.add_command(label="保存游戏", command=on_save_game)#添加保存游戏命令
        game_menu.add_command(label="加载游戏", command=on_load_game)#添加加载游戏命令
        game_menu.add_separator()#添加分隔符
        game_menu.add_command(label="退出", command=root.quit)#添加退出命令
        self.menubar.add_cascade(label="游戏", menu=game_menu)#添加游戏菜单
        
        # 帮助菜单
        help_menu = tk.Menu(self.menubar, tearoff=0)#创建帮助菜单
        help_menu.add_command(label="规则", command=self.show_rules)#添加规则命令
        self.menubar.add_cascade(label="帮助", menu=help_menu)#添加帮助菜单
        
        root.config(menu=self.menubar)#配置根窗口菜单

    def show_rules(self):#游戏菜单组件显示规则
        rules = """
围棋规则：
1. 玩家轮流在棋盘的交叉点上落子
2. 当棋子没有气（相邻的空点）时会被提走
3. 当双方连续放弃落子时游戏结束
4. 拥有更多领地的玩家获胜

五子棋规则：
1. 玩家轮流在棋盘的交叉点上落子
2. 首先在横、竖或斜线上连成五子的玩家获胜
3. 如果棋盘填满且无人获胜，则为平局
        """
        messagebox.showinfo("游戏规则", rules)#显示游戏规则

class WelcomeScreen:#欢迎界面组件
    """欢迎界面组件"""
    def __init__(self, parent, on_new_game, on_load_game):#欢迎界面组件初始化
        self.frame = tk.Frame(parent)#创建框架
        self.frame.pack(expand=True, fill='both')#设置框架大小
        
        title_label = tk.Label(self.frame, text="欢迎来到围棋和五子棋!", font=("宋体", 24))#创建标题标签
        title_label.pack(pady=40)#设置标题标签位置
        
        new_game_btn = tk.Button(self.frame, text="新建游戏", command=on_new_game, 
                                font=("宋体", 14), width=15, height=2)#创建新建游戏按钮
        new_game_btn.pack(pady=20)#设置新建游戏按钮位置
        
        load_game_btn = tk.Button(self.frame, text="加载游戏", command=on_load_game, 
                                 font=("宋体", 14), width=15, height=2)#创建加载游戏按钮
        load_game_btn.pack(pady=20)#设置加载游戏按钮位置

    def show(self):#欢迎界面组件显示
        self.frame.pack(expand=True, fill='both')#设置框架大小

    def hide(self):#欢迎界面组件隐藏
        self.frame.pack_forget()#隐藏框架

class NewGameDialog:#新建游戏对话框组件
    """新建游戏对话框组件"""
    def __init__(self, parent, on_start_game): #新建游戏对话框组件初始化
        self.dialog = tk.Toplevel(parent)#创建对话框
        self.dialog.title("新建游戏")#设置对话框标题
        self.dialog.geometry("400x300")#设置对话框大小
        self.dialog.transient(parent)#设置对话框父窗口
        self.dialog.grab_set()#设置对话框焦点
        
        tk.Label(self.dialog, text="游戏类型:", font=("宋体", 12)).pack(pady=10)#创建游戏类型标签
        self.game_type_var = tk.StringVar(value="围棋")#创建游戏类型变量
        game_type_frame = tk.Frame(self.dialog)#创建游戏类型框架
        game_type_frame.pack()#设置游戏类型框架位置
        tk.Radiobutton(game_type_frame, text="围棋", variable=self.game_type_var, 
                      value="围棋", font=("宋体", 12)).pack(side=tk.LEFT, padx=20)#创建围棋单选按钮
        tk.Radiobutton(game_type_frame, text="五子棋", variable=self.game_type_var, 
                      value="五子棋", font=("宋体", 12)).pack(side=tk.LEFT, padx=20)#创建五子棋单选按钮
        
        tk.Label(self.dialog, text="棋盘大小:", font=("宋体", 12)).pack(pady=10)#创建棋盘大小标签
        self.board_size_var = tk.StringVar(value="19")#创建棋盘大小变量
        board_size_entry = tk.Entry(self.dialog, textvariable=self.board_size_var, 
                                  font=("宋体", 12))#创建棋盘大小输入框
        board_size_entry.pack(pady=10)#设置棋盘大小输入框位置
        
        start_btn = tk.Button(self.dialog, text="开始游戏", 
                            command=lambda: self._start_game(on_start_game), 
                            font=("宋体", 14), width=15, height=2)#创建开始游戏按钮
        start_btn.pack(pady=20)#设置开始游戏按钮位置

    def _start_game(self, on_start_game):#新建游戏对话框组件开始游戏
        try:
            game_type = GameType(self.game_type_var.get())#获取游戏类型
            board_size = int(self.board_size_var.get())#获取棋盘大小
            on_start_game(game_type, board_size)#开始游戏
            self.dialog.destroy()#销毁对话框
        except InvalidBoardSizeError as e:#新建游戏对话框组件处理错误
            messagebox.showerror("错误", str(e))#显示错误信息
        except ValueError:#新建游戏对话框组件处理错误
            messagebox.showerror("错误", "请输入有效的棋盘大小")#显示错误信息

class GameGUI:#游戏主界面
    """游戏主界面"""
    def __init__(self):#游戏主界面初始化
        self.root = tk.Tk()#创建根窗口
        self.root.title("围棋和五子棋")#设置根窗口标题
        self.root.geometry("900x900")#设置根窗口大小
        self.root.minsize(900, 900)#设置根窗口最小大小
        
        self.current_game_type = None#当前游戏类型
        self.current_board_size = 0#当前棋盘大小
        self.client = None#当前客户端
        
        # 创建菜单和欢迎界面
        self.menu = GameMenu(self.root, self.show_new_game_dialog, self.save_game, self.load_game)#创建菜单
        self.welcome_screen = WelcomeScreen(self.root, self.show_new_game_dialog, self.load_game)#创建欢迎界面
        
        self.root.option_add("*Font", ("宋体", 12))#设置根窗口字体

    def show_new_game_dialog(self):#游戏主界面显示新建游戏对话框
        NewGameDialog(self.root, self.start_new_game)#显示新建游戏对话框

    def start_new_game(self, game_type: GameType, board_size: int):#游戏主界面开始新游戏
        self.current_game_type = game_type#设置当前游戏类型
        self.current_board_size = board_size#设置当前棋盘大小
        
        self.welcome_screen.hide()#隐藏欢迎界面
        
        if self.client:
            self.client.frame.destroy()#销毁客户端框架
            self.client = None#设置当前客户端为None
        
        # 创建对应的后端
        if game_type == GameType.GO:#创建围棋后端
            backend = GoBackend(board_size)
        else:#创建五子棋后端
            backend = GomokuBackend(board_size)
        
        # 创建客户端
        self.client = GUIClient(backend, self.root)#创建客户端
        
        # 将客户端注册为观察者
        backend.attach(self.client)#将客户端注册为观察者

    def save_game(self):#游戏主界面保存游戏
        if not self.client or not self.client.backend:#游戏主界面保存游戏
            messagebox.showerror("错误", "没有正在进行的游戏")#显示错误信息
            return#返回
            
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",#设置默认文件扩展名
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],#设置文件类型
                title="保存游戏"#设置对话框标题
            )
            if filename:
                state = self.client.backend.get_game_state()#获取游戏状态
                game_data = {
                    'board_size': self.current_board_size,
                    'game_type': self.current_game_type.value,
                    'board': [[stone.value for stone in row] for row in state.board],
                    'current_player': state.current_player.value,
                    'move_history': state.move_history,
                    'game_over': state.game_over,
                    'winner': state.winner.value if state.winner else None,
                    'resigned': state.resigned,
                    'pass_count': state.pass_count,
                    'undo_count': state.undo_count
                }
                with open(filename, 'w') as f:
                    json.dump(game_data, f)#保存游戏数据
                messagebox.showinfo("成功", "游戏保存成功")#显示保存成功信息
        except Exception as e:#游戏主界面保存游戏处理错误
            messagebox.showerror("保存失败", str(e))#显示保存失败信息

    def load_game(self):#游戏主界面加载游戏
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],#设置文件类型
                title="加载游戏"#设置对话框标题
            )
            if filename:
                with open(filename, 'r') as f:
                    game_data = json.load(f)#加载游戏数据
                
                game_type = GameType(game_data['game_type'])#获取游戏类型
                board_size = game_data['board_size']#获取棋盘大小
                
                # 创建对应的后端
                if game_type == GameType.GO:#创建围棋后端
                    backend = GoBackend(board_size)
                else:#创建五子棋后端
                    backend = GomokuBackend(board_size)
                
                # 恢复游戏状态
                state = backend.get_game_state()#获取游戏状态
                state.board = [[StoneColor(value) for value in row] for row in game_data['board']]#设置棋盘
                state.current_player = StoneColor(game_data['current_player'])#设置当前玩家
                state.move_history = game_data['move_history']#设置落子历史
                state.game_over = game_data['game_over']#设置游戏结束
                state.winner = StoneColor(game_data['winner']) if game_data['winner'] else None#设置胜利者
                state.resigned = game_data['resigned']#设置认输
                state.pass_count = game_data.get('pass_count', 0)#设置认输次数
                state.undo_count = game_data.get('undo_count', 0)#设置悔棋次数
                
                # 更新当前游戏信息
                self.current_game_type = game_type#设置当前游戏类型
                self.current_board_size = board_size#设置当前棋盘大小
                
                # 创建新的客户端
                if self.client:#创建新的客户端
                    self.client.frame.destroy()#销毁客户端框架
                self.client = GUIClient(backend, self.root)#创建新的客户端
                backend.attach(self.client)#将客户端注册为观察者
                
                messagebox.showinfo("成功", "游戏加载成功")#显示加载成功信息
        except Exception as e:#游戏主界面加载游戏处理错误
            messagebox.showerror("加载失败", str(e))#显示加载失败信息

    def run(self):#游戏主界面运行
        self.root.mainloop()#运行主窗口

if __name__ == "__main__":#游戏主界面运行
    interface = GameGUI()#创建游戏主界面
    interface.run()#运行游戏主界面
