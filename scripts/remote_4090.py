# remote_4090.py — 从开发机(KasoForum)连 4090(kasopc, 192.168.137.2) 的 paramiko 助手
# 背景: 4090 内置 OpenSSH 9.5 对公钥登录有系统级 bug(任何账号都在 Postponed publickey 后崩溃),
#       密码路径未受波及, 故用 ninfer 账号 + 密码登录 (密码 = mkuser_4090.ps1 中 $NewPass)。
# 用法:
#   py -3 misc\remote_4090.py test
#   py -3 misc\remote_4090.py run "<remote command (cmd.exe 语法)>"
#   py -3 misc\remote_4090.py put <local> <remote>
#   py -3 misc\remote_4090.py get <remote> <local>
import sys, io, paramiko

def _reconfigure_stdio():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

_reconfigure_stdio()

HOST = '192.168.137.2'
PORT = 22
USER = 'ninfer'
PASS = 'NinFer-Build2026-x7'

def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS,
              allow_agent=False, look_for_keys=False, timeout=15, banner_timeout=30, auth_timeout=30)
    return c

def run(c, cmd, timeout=600):  # timeout=None => wait indefinitely (long builds)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    rc = stdout.channel.recv_exit_status()
    if out:
        sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    if rc != 0:
        sys.stderr.write(f'[exit code: {rc}]\n')
    return rc

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'test'
    c = connect()
    if mode == 'test':
        rc = run(c, 'echo AUTH_OK_NINFER_PW & whoami & ver')
    elif mode == 'run':
        t = sys.argv[3] if len(sys.argv) > 3 else '600'
        rc = run(c, sys.argv[2], None if t == 'inf' else int(t))
    elif mode == 'put':
        s = c.open_sftp()
        s.put(sys.argv[2], sys.argv[3])
        print('PUT OK:', sys.argv[2], '->', sys.argv[3])
        rc = 0
    elif mode == 'get':
        s = c.open_sftp()
        s.get(sys.argv[2], sys.argv[3])
        print('GET OK:', sys.argv[2], '->', sys.argv[3])
        rc = 0
    else:
        print('unknown mode', file=sys.stderr)
        rc = 2
    c.close()
    sys.exit(rc)

if __name__ == '__main__':
    main()