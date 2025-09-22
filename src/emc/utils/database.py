import pymysql


def create_optuna_db(db_name, user="optuna_user", password="123"):
    # Connect via Unix socket as root (no password needed with unix_socket auth)
    conn = pymysql.connect(
        unix_socket="/var/run/mysqld/mysqld.sock",
        user="root",
        password=""  # Leave empty for unix_socket auth
    )
    cur = conn.cursor()

    # SQL commands
    sql_commands = f"""
    DROP DATABASE IF EXISTS {db_name};
    CREATE DATABASE {db_name};
    CREATE USER IF NOT EXISTS '{user}'@'%' IDENTIFIED BY '{password}';
    GRANT ALL PRIVILEGES ON {db_name}.* TO '{user}'@'%';
    FLUSH PRIVILEGES;
    """

    # Execute each statement separately
    for cmd in sql_commands.strip().split(";"):
        if cmd.strip():
            cur.execute(cmd)

    conn.commit()
    cur.close()
    conn.close()
