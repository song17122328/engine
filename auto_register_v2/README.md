MyFlaskProject/

├── app/                  # 主应用包
│   ├── __init__.py       # 初始化Flask应用
│   ├── models.py         # 数据库模型
│   ├── routes.py         # 路由定义
│   ├── services/         # 业务逻辑
│   │   ├── __init__.py
│   │   ├── cookie_service.py
│   │   ├── proxy_service.py
│   │   └── submission_service.py
│   └── templates/        # 模板文件
│       └── index.html
├── config.py             # 配置文件
└── run.py                # 启动脚本