"""
SMPL模型转换脚本
将包含chumpy对象的pickle文件转换为纯numpy格式
"""

import pickle
import numpy as np
import sys

# 添加chumpy包路径
sys.path.insert(0, '.')

def convert_smpl_model(input_path, output_path):
    """转换SMPL模型文件"""
    print(f"正在加载模型：{input_path}")
    
    # 加载原始模型
    with open(input_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # 转换函数
    def to_numpy(obj):
        if hasattr(obj, 'r'):
            result = obj.r()
            if hasattr(result, 'toarray'):
                return result.toarray()
            return to_numpy(result)
        elif isinstance(obj, np.ndarray):
            if obj.dtype == np.object_:
                return np.array([to_numpy(item) for item in obj])
            return obj
        else:
            return np.array(obj)
    
    # 转换所有数据
    converted_data = {}
    for key, value in data.items():
        print(f"转换 {key}...")
        try:
            converted_data[key] = to_numpy(value)
        except Exception as e:
            print(f"警告：无法转换 {key}: {e}")
            converted_data[key] = value
    
    # 保存转换后的模型
    print(f"正在保存转换后的模型：{output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(converted_data, f)
    
    print("转换完成！")

if __name__ == "__main__":
    input_path = 'smpl/SMPL_NEUTRAL.pkl'
    output_path = 'smpl/SMPL_NEUTRAL_converted.pkl'
    convert_smpl_model(input_path, output_path)