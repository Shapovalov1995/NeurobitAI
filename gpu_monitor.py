import torch

class GPUMonitor:
    def __init__(self):
        """Мониторинг состояния GPU"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_stats(self):
        """Возвращает статистику по GPU"""
        stats = {
            "device": self.device,
            "memory_used": "N/A",
            "memory_total": "N/A"
        }
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            stats.update({
                "device": props.name,
                "memory_used": f"{torch.cuda.memory_allocated() / 1024**2:.1f}",
                "memory_total": f"{props.total_memory / 1024**2:.1f}"
            })
        
        return stats
    
    def print_stats(self):
        """Выводит текущую статистику GPU"""
        stats = self.get_stats()
        print(f"\nGPU Status: {stats['device']}")
        print(f"Memory: {stats['memory_used']}/{stats['memory_total']} MB")