"""
工具调用模块 - Function Calling
支持音量调节等系统工具
"""
import logging
from typing import Dict, Any, List, Callable

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册中心"""
    
    def __init__(self):
        self.tools: Dict[str, Dict] = {}
        self.handlers: Dict[str, Callable] = {}
    
    def register(self, name: str, description: str, parameters: Dict, handler: Callable):
        """注册工具"""
        self.tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        self.handlers[name] = handler
        logger.debug(f"注册工具: {name}")
    
    def get_tools(self) -> List[Dict]:
        """获取所有工具定义"""
        return list(self.tools.values())
    
    def execute(self, name: str, arguments: Dict) -> Any:
        """执行工具"""
        if name not in self.handlers:
            logger.error(f"未知工具: {name}")
            return {"error": f"未知工具: {name}"}
        
        try:
            result = self.handlers[name](**arguments)
            logger.info(f"执行工具 {name}: {arguments} -> {result}")
            return result
        except Exception as e:
            logger.error(f"执行工具失败 {name}: {e}")
            return {"error": str(e)}


class VolumeTool:
    """音量控制工具"""
    
    @staticmethod
    def get_tool_definition():
        """获取工具定义"""
        return {
            "type": "function",
            "function": {
                "name": "set_volume",
                "description": "调节系统音量。可以设置音量大小、静音或取消静音。当用户说'把音量调大一点'、'静音'、'音量调到50%'等指令时使用此工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["set", "increase", "decrease", "mute", "unmute", "get"],
                            "description": "音量操作类型: set=设置具体值, increase=增加音量, decrease=降低音量, mute=静音, unmute=取消静音, get=获取当前音量"
                        },
                        "value": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "音量值(0-100)，仅在action为set/increase/decrease时使用"
                        }
                    },
                    "required": ["action"]
                }
            }
        }
    
    @staticmethod
    def execute(action: str, value: int = None) -> Dict[str, Any]:
        """
        执行音量控制
        
        Args:
            action: 操作类型
            value: 音量值
            
        Returns:
            执行结果
        """
        try:
            import subprocess
            import re
            
            def get_current_volume():
                result = subprocess.run(['amixer', 'get', 'Master'], 
                                        capture_output=True, text=True)
                match = re.search(r'\[(\d+)%\]', result.stdout)
                return int(match.group(1)) if match else 50
            
            def set_volume(vol):
                subprocess.run(['amixer', 'set', 'Master', f'{vol}%'], 
                              capture_output=True, check=True)
                return vol
            
            if action == "get":
                current = get_current_volume()
                return {"success": True, "volume": current, "message": f"当前音量是 {current}%"}
            
            elif action == "set":
                if value is None:
                    return {"success": False, "error": "需要指定音量值"}
                value = max(0, min(100, value))
                set_volume(value)
                return {"success": True, "volume": value, "message": f"音量已设置为 {value}%"}
            
            elif action == "increase":
                current = get_current_volume()
                delta = value if value else 10
                new_vol = min(100, current + delta)
                set_volume(new_vol)
                return {"success": True, "volume": new_vol, "message": f"音量已增加到 {new_vol}%"}
            
            elif action == "decrease":
                current = get_current_volume()
                delta = value if value else 10
                new_vol = max(0, current - delta)
                set_volume(new_vol)
                return {"success": True, "volume": new_vol, "message": f"音量已降低到 {new_vol}%"}
            
            elif action == "mute":
                subprocess.run(['amixer', 'set', 'Master', 'mute'], capture_output=True)
                return {"success": True, "muted": True, "message": "已静音"}
            
            elif action == "unmute":
                subprocess.run(['amixer', 'set', 'Master', 'unmute'], capture_output=True)
                return {"success": True, "muted": False, "message": "已取消静音"}
            
            else:
                return {"success": False, "error": f"未知操作: {action}"}
                
        except Exception as e:
            logger.error(f"音量控制失败: {e}")
            return {"success": False, "error": str(e)}


class SystemTools:
    """系统工具集合"""
    
    def __init__(self):
        self.registry = ToolRegistry()
        self._register_tools()
    
    def _register_tools(self):
        """注册所有工具"""
        self.registry.register(
            name="set_volume",
            description="调节系统音量",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set", "increase", "decrease", "mute", "unmute", "get"]
                    },
                    "value": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": ["action"]
            },
            handler=VolumeTool.execute
        )
    
    def get_tools(self) -> List[Dict]:
        """获取所有工具定义"""
        return self.registry.get_tools()
    
    def execute(self, name: str, arguments: Dict) -> Any:
        """执行工具"""
        return self.registry.execute(name, arguments)


def get_default_tools() -> List[Dict]:
    """获取默认工具列表"""
    return [
        VolumeTool.get_tool_definition(),
    ]


def execute_tool(name: str, arguments: Dict) -> Dict[str, Any]:
    """执行工具（简化接口）"""
    if name == "set_volume":
        return VolumeTool.execute(**arguments)
    return {"error": f"未知工具: {name}"}
