import glob
import json
import os

import cv2
import numpy as np

PASTA_IMAGENS = "imagens"
PASTA_SAIDA = "resultados"
TAM_QUADRADO = 0.09
PADRAO = (9, 4)
ESCALAS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5]
MAX_RESIDUO_HOMOGRAFIA = 6.0
FATOR_OUTLIER = 3.0

FLAGS_CALIB = cv2.CALIB_FIX_K3


FLAGS_LEITURA = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


def pontos_objeto(padrao, tam=TAM_QUADRADO):
    cols, lins = padrao
    obj = np.zeros((cols * lins, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:lins].T.reshape(-1, 2) * tam
    return obj


def residuo_homografia(cantos, padrao=PADRAO):
    grade = pontos_objeto(padrao)[:, :2]
    c = cantos.reshape(-1, 2)
    H, _ = cv2.findHomography(grade, c)
    return float(np.linalg.norm(cv2.perspectiveTransform(grade[None], H)[0] - c, axis=1).mean())


def detectar_cantos(cinza):

    n = PADRAO[0] * PADRAO[1]
    detectores = [
        lambda im: cv2.findChessboardCornersSB(im, PADRAO, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY),
        lambda im: cv2.findChessboardCornersSB(im, PADRAO, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY),
        lambda im: cv2.findChessboardCorners(im, PADRAO, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE),
    ]
    melhor, melhor_res = None, np.inf
    for esc in ESCALAS:
        peq = cv2.resize(cinza, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
        for det in detectores:
            ok, cantos = det(peq)
            if not ok or len(cantos) != n:
                continue
            cantos = ((cantos + 0.5) / esc - 0.5).astype(np.float32)  # volta à resolução cheia
            res = residuo_homografia(cantos)
            if res < melhor_res:
                melhor, melhor_res = cantos, res
    if melhor is None or melhor_res > MAX_RESIDUO_HOMOGRAFIA:
        return None, melhor_res
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    melhor = cv2.cornerSubPix(cinza, melhor, (11, 11), (-1, -1), crit)
    return melhor, residuo_homografia(melhor)


def erro_reprojecao(obj, img_pts, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    d = np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)
    return proj.reshape(-1, 2), d


def residuo_retilinidade(pts, padrao):

    cols, lins = padrao
    grade = pts.reshape(lins, cols, 2)
    res = []
    for linha in list(grade) + list(grade.transpose(1, 0, 2)):
        vx, vy, x0, y0 = cv2.fitLine(linha.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        n = np.array([-vy, vx])
        res.extend(((linha - [x0, y0]) @ n).tolist())
    return float(np.sqrt(np.mean(np.square(res))))


def desenhar_eixos(img, rvec, tvec, K, dist, comprimento, espessura):
    pts3d = np.float32([[0, 0, 0], [comprimento, 0, 0], [0, comprimento, 0], [0, 0, -comprimento]])
    pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    pts2d = np.int32(pts2d.reshape(-1, 2))
    origem = tuple(pts2d[0])
    cv2.line(img, origem, tuple(pts2d[1]), (128, 0, 128), espessura)
    cv2.line(img, origem, tuple(pts2d[2]), (203, 192, 255), espessura)
    cv2.line(img, origem, tuple(pts2d[3]), (0, 0, 255), espessura)
    return img


def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    for sub in ("deteccao", "sem_distorcao", "projecao"):
        os.makedirs(os.path.join(PASTA_SAIDA, sub), exist_ok=True)

    arquivos = sorted(glob.glob(os.path.join(PASTA_IMAGENS, "*.jpg")))
    obj_pts, img_pts, nomes, padroes = [], [], [], []
    tam_img = None

    print("detecção dos cantos")
    for f in arquivos:
        img = cv2.imread(f, FLAGS_LEITURA)
        cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tam_img = cinza.shape[::-1]
        cantos, res = detectar_cantos(cinza)
        nome = os.path.basename(f)
        if cantos is None:
            print(f"  {nome}: tabuleiro NÃO encontrado (descartada)")
            continue
        padrao = PADRAO
        print(f"  {nome}: {padrao[0]}x{padrao[1]} cantos | resíduo homografia {res:.2f}px")
        obj_pts.append(pontos_objeto(padrao))
        img_pts.append(cantos)
        nomes.append(nome)
        padroes.append(padrao)
        vis = cv2.drawChessboardCorners(img.copy(), padrao, cantos, True)
        cv2.imwrite(os.path.join(PASTA_SAIDA, "deteccao", nome), cv2.resize(vis, None, fx=0.3, fy=0.3))

    print("\ncalibração")
    descartadas_outlier = []
    while True:
        rms, K, dist, rvecs, tvecs, _, _, erros_view = cv2.calibrateCameraExtended(
            obj_pts, img_pts, tam_img, None, None, flags=FLAGS_CALIB)
        e = erros_view.ravel()
        pior = int(np.argmax(e))
        if e[pior] <= FATOR_OUTLIER * np.median(e):
            break
        print(f"  {nomes[pior]}: erro {e[pior]:.2f}px > {FATOR_OUTLIER}x mediana -> descartada, recalibrando")
        descartadas_outlier.append(nomes[pior])
        for lst in (obj_pts, img_pts, nomes, padroes):
            lst.pop(pior)
    print(f"  Imagens usadas: {len(nomes)} | tamanho: {tam_img[0]}x{tam_img[1]}")
    print(f"  Erro RMS de reprojeção: {rms:.3f} px")
    print("  K =\n", np.array2string(K, precision=2, suppress_small=True))
    print("  dist (k1 k2 p1 p2 k3) =", np.array2string(dist.ravel(), precision=4))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fov_x = 2 * np.degrees(np.arctan(tam_img[0] / (2 * fx)))
    fov_y = 2 * np.degrees(np.arctan(tam_img[1] / (2 * fy)))
    print(f"  FOV horizontal = {fov_x:.1f}°, vertical = {fov_y:.1f}°")

    np.savez(os.path.join(PASTA_SAIDA, "calibracao.npz"), K=K, dist=dist, rms=rms,
             rvecs=np.array(rvecs), tvecs=np.array(tvecs), nomes=np.array(nomes))
    fs = cv2.FileStorage(os.path.join(PASTA_SAIDA, "calibracao.yaml"), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", tam_img[0]); fs.write("image_height", tam_img[1])
    fs.write("camera_matrix", K); fs.write("dist_coeffs", dist); fs.write("rms", rms)
    fs.release()

    por_imagem = []
    for i, nome in enumerate(nomes):
        R, _ = cv2.Rodrigues(rvecs[i])
        centro_cam = (-R.T @ tvecs[i]).ravel()
        por_imagem.append({
            "imagem": nome, "padrao": list(padroes[i]),
            "erro_rms_px": float(erros_view[i, 0]),
            "distancia_camera_m": float(np.linalg.norm(tvecs[i])),
            "centro_camera_m": centro_cam.round(3).tolist(),
            "R": R.round(4).tolist(), "t_m": tvecs[i].ravel().round(4).tolist(),
        })
       


    K_alpha0, _ = cv2.getOptimalNewCameraMatrix(K, dist, tam_img, 0, tam_img)
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K_alpha0, tam_img, cv2.CV_32FC1)
    retilinidade = []
    for i, nome in enumerate(nomes):
        img = cv2.imread(os.path.join(PASTA_IMAGENS, nome), FLAGS_LEITURA)
        und = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)       # alpha=0
        cinza_o = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cinza_u = cv2.cvtColor(cv2.undistort(img, K, dist), cv2.COLOR_BGR2GRAY)  # mesma K
        sobrepos = cv2.merge([cinza_u, cinza_u, cinza_o])
        pts_und = cv2.undistortPoints(img_pts[i], K, dist, P=K).reshape(-1, 2)
        antes = residuo_retilinidade(img_pts[i].reshape(-1, 2), padroes[i])
        depois = residuo_retilinidade(pts_und, padroes[i])
        retilinidade.append({"imagem": nome, "antes_px": antes, "depois_px": depois})
   

        esc = 0.25
        a = cv2.resize(img, None, fx=esc, fy=esc)
        b = cv2.resize(und, None, fx=esc, fy=esc)
        c = cv2.resize(sobrepos, None, fx=esc, fy=esc)
        for im, txt in ((a, "original"), (b, "sem distorcao (alpha=0)"), (c, "vermelho=original ciano=corrigida")):
            cv2.putText(im, txt, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(PASTA_SAIDA, "sem_distorcao", nome), np.hstack([a, b, c]))

    mx, my = cv2.initUndistortRectifyMap(K, dist, None, K, tam_img, cv2.CV_32FC1)
    desloc = np.hypot(mx - np.arange(tam_img[0])[None], my - np.arange(tam_img[1])[:, None])
    casco = cv2.convexHull(np.vstack([p.reshape(-1, 2) for p in img_pts]).astype(np.float32))
    mascara = np.zeros(desloc.shape, np.uint8)
    cv2.fillConvexPoly(mascara, casco.astype(np.int32), 1)
    desloc_util = float(desloc[mascara > 0].max())
    print(f"  Deslocamento máximo na região coberta pelos cantos: {desloc_util:.1f} px "
          f"(cantos da imagem, extrapolado: {desloc.max():.1f} px)")
    mapa = cv2.applyColorMap(np.uint8(np.clip(desloc / desloc_util, 0, 1) * 255), cv2.COLORMAP_JET)
    cv2.polylines(mapa, [casco.astype(np.int32)], True, (255, 255, 255), 12)
    for p in img_pts:
        for q in p.reshape(-1, 2):
            cv2.circle(mapa, tuple(np.int32(q)), 12, (255, 255, 255), -1)
    cv2.imwrite(os.path.join(PASTA_SAIDA, "mapa_distorcao.jpg"), cv2.resize(mapa, None, fx=0.25, fy=0.25))



    projecao = []
    for i, nome in enumerate(nomes):
        obj = obj_pts[i]
        proj, d = erro_reprojecao(obj, img_pts[i], rvecs[i], tvecs[i], K, dist)

        projecao.append({
            "imagem": nome,
            "erro_medio_px": float(d.mean()),
            "erro_max_px": float(d.max()),
        })


        img = cv2.imread(os.path.join(PASTA_IMAGENS, nome), FLAGS_LEITURA)
        r = max(6, img.shape[1] // 250)
        desenhar_eixos(img, rvecs[i], tvecs[i], K, dist, 3 * TAM_QUADRADO, max(4, r))
        cv2.imwrite(os.path.join(PASTA_SAIDA, "projecao", nome), cv2.resize(img, None, fx=0.3, fy=0.3))

    resumo = {
        "tabuleiro": {"quadrado_m": TAM_QUADRADO, "cantos_internos": list(PADRAO)},
        "tamanho_imagem": list(tam_img),
        "imagens_usadas": nomes,
        "imagens_descartadas": [os.path.basename(f) for f in arquivos if os.path.basename(f) not in nomes],
        "descartadas_por_erro_alto": descartadas_outlier,
        "rms_px": rms, "K": K.tolist(), "dist": dist.ravel().tolist(),
        "K_otima_alpha0": K_alpha0.tolist(),
        "fov_graus": [fov_x, fov_y],
        "deslocamento_max_distorcao_px_regiao_util": desloc_util,
        "deslocamento_max_distorcao_px_imagem_toda": float(desloc.max()),
        "por_imagem": por_imagem, "retilinidade": retilinidade, "projecao": projecao,
    }
    with open(os.path.join(PASTA_SAIDA, "resumo.json"), "w") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print(f"\nresultados salvos em ./{PASTA_SAIDA}/")


if __name__ == "__main__":
    main()
