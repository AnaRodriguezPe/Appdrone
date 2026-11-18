import csv
import json
import os
import socket
import threading
import time
import numpy as np

# data de kivy TCP
HOST_IN = "0.0.0.0"
PORT_IN = 65432

# data para flair UDP
DESTINO = ("172.26.209.106", 65433)
PERIODO = 0.1

# limites
X_MIN, X_MAX = -3, 3
Z_MIN, Z_MAX = 0.8, 3

# guardar para malab
GUARDAR_CSV = True
CARPETA = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_CSV = os.path.join(CARPETA, "line.csv")
ARCHIVO_KALMAN_CSV = os.path.join(CARPETA, "line2.csv")

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
trayectoria = 0


def write_csv_raw(path, t, x_m, z_m, meta):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        f.write(f"# width_px={meta['width_px']},height_px={meta['height_px']},dpi={meta['dpi']}\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["t", "x_m", "z_m"])
        for ti, xm, zm in zip(t, x_m, z_m):
            w.writerow([f"{ti:.6f}", f"{xm:.4f}", f"{zm:.4f}"])
    os.replace(tmp, path)

def write_csv_kalman(path,t, x_m, z_m, vx_m, vz_m, ax_m, az_m, meta):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        f.write(f"# width_px={meta['width_px']},height_px={meta['height_px']},dpi={meta['dpi']}\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["t", "x_m", "z_m", "vx_m", "vz_m", "ax_m", "az_m"])
        for ti, xm, zm, vxi, vzi, axi, azi in zip(t, x_m, z_m, vx_m, vz_m, ax_m, az_m):
            w.writerow([f"{ti:.6f}", f"{xm:.4f}", f"{zm:.4f}", f"{vxi:.4f}", f"{vzi:.4f}", f"{axi:.4f}", f"{azi:.4f}"])
    os.replace(tmp, path)

#def send_data(x_m, z_m, meta, mi_id):
def send_data(xf, zf, meta, mi_id):
    ancho_m = meta["ancho_util_px"] / meta["dpi"] * 0.0254
    alto_m = meta["alto_util_px"] / meta["dpi"] * 0.0254

    for x, z in zip(xf, zf):
        if mi_id != trayectoria:
            return
        nx = max(0, min(1, x / ancho_m))
        nz = max(0, min(1, z / alto_m))
        x_vuelo = X_MIN + nx * (X_MAX - X_MIN)
        z_vuelo = Z_MIN + nz * (Z_MAX - Z_MIN)
        udp.sendto(f"{x_vuelo:.4f} {z_vuelo:.4f}".encode(), DESTINO)
        time.sleep(PERIODO)


def recibir_datos():
    global trayectoria
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST_IN, PORT_IN))
        s.listen()
        s.settimeout(1.0)

        while True:
            try:
                conn, addr = s.accept()
            except socket.timeout:
                continue

            with conn:
                conn.settimeout(5.0)
                try:
                    buffer = b""
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        buffer += chunk

                    datos = json.loads(buffer.decode("utf-8"))
                    meta = {k: float(v) for k, v in datos["meta"].items()}
                    t, x_m, z_m = zip(*datos["sample"])

                    filtrado = kalman(list(zip(t, x_m, z_m)))

                    if GUARDAR_CSV:
                        write_csv_raw(ARCHIVO_CSV, t, x_m, z_m, meta)

                        tf, xf, zf, vxf, vzf, axf,azf = zip(*filtrado)
                        write_csv_kalman(ARCHIVO_KALMAN_CSV, tf, xf, zf, vxf, vzf, axf,azf, meta)

                    filtrado_para_app = [(ti, xi, zi) for ti, xi, zi, _, _, _, _ in filtrado]
                    conn.sendall((json.dumps({"filtrado": filtrado_para_app}) + "\n").encode("utf-8"))
                    #conn.sendall((json.dumps({"filtrado": filtrado}) + "\n").encode("utf-8"))

                    trayectoria += 1
                    threading.Thread(target=send_data,
                                     args=(xf, zf, meta, trayectoria),
                                     #args=(x_m, z_m, meta, trayectoria),
                                     daemon=True).start()
                    print("recibido")
                    #print(Ks[-1])

                except Exception as e:
                    print(f"Error: {e}")



Ks=[]
Ps=[]
#matplotlib si esta muy ruidoso mas caso al modelo

def kalman(datam):
    global Ks, Ps
    r = 0.08
    s = 0.5
    sv = 10
    sa=100

    H = np.array([[1, 0, 0, 0,0,0], [0, 1, 0, 0,0,0]], dtype=float)
    Q = np.diag([s**2, s**2, sv**2, sv**2, sa**2, sa**2])
    R = np.diag([r**2, r**2])
    I = np.eye(6)

    t0, x0, z0 = datam[0]
    X = np.array([[x0], [z0], [0], [0],[0], [0]], dtype=float)
    P = np.diag([r**2, r**2, sv**2, sv**2, sa**2, sa**2])

    salida = [(t0, x0, z0, float(X[2, 0]), float(X[3, 0]), float(X[4, 0]), float(X[5, 0]))]
    prev_tk = t0

    for k in range(1, len(datam)):
        tk, xk, zk = datam[k]
        #dt = max(tk - datam[k - 1][0], 1e-6)
        dt = max(tk - prev_tk, 1e-6)
        prev_tk = tk
        A = np.array([[1, 0, dt, 0, 0.5*dt**2, 0], 
                      [0, 1, 0, dt, 0, 0.5*dt**2], 
                      [0, 0, 1, 0, dt, 0], 
                      [0, 0, 0, 1, 0, dt],
                      [0, 0, 0, 0, 1, 0], 
                      [0, 0, 0, 0, 0, 1]], dtype=float)
        Z=np.array([[xk], [zk]])

        # prediccion
        X = A @ X #np.dot(A, X)
        P = (A @ P @ A.T)+Q #np.dot(A, np.dot(P, A.T)) + Q

        S =(H @ P @ H.T)+ R    #np.dot(H, np.dot(P, H.T)) + R
        K =P @ H.T @ np.linalg.inv(S)  #np.dot(P,np.dot(H.T, np.linalg.inv(S)))
        Ks.append(K)
        X = X + K @ (Z - H @ X) #X + np.dot(K, Z - np.dot(H, X))
        P = (I - K @ H) @ P #np.dot((I - np.dot(K, H)), P)
        Ps.append(P)

        salida.append((tk, float(X[0, 0]), float(X[1, 0]), float(X[2, 0]), float(X[3, 0]), float(X[4, 0]), float(X[5, 0])))
    return salida


if __name__ == "__main__":
    try:
        recibir_datos()
    except KeyboardInterrupt:
        print("\ncerrar")
