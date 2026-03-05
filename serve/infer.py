import logging
import time 
import json

import os
# 在导入ollama前清除代理环境变量
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('all_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('ALL_PROXY', None)


import ollama



def run(prompt):
    start = time.time()

    try:
        # provider - Ollama非流式调用
        response = ollama.chat(
            model='qwen3.5:27b',  # 或你下载的其他模型，如 'qwen2.5', 'mistral' 等
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )
        
        result = {
            "answer": response['message']['content']
        }
    except Exception as e:
        logging.error(f"Ollama调用失败: {str(e)}")
        result = {
            "error": str(e),
            "answer": None
        }
    
    latency = time.time() - start
    logging.info({
        "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
        "latency": latency,
        "success": "error" not in result
    })
    
    return result


if __name__ == "__main__":
    import sys
    result = run(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False))