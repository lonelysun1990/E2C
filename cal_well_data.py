import numpy as np


def load_pvdo():
    data_file = '/data3/Astro/personal/zjin/sim_runs/case8_9w_bhp_rate_ms_gau/adgprs/template/'
    pvdo_table = np.loadtxt(data_file + 'PVDO.DAT', skiprows=1, comments='/')
    return pvdo_table


def cal_well_index(k):
    dx = dy = 164.042
    dz = 32.81
    r = 0.5
    alpha = 0.001127
#     r0_ = 0.28  *(pow((pow(ky / kx, 0.5)*dx*dx + pow(kx / ky, 0.5)*dy*dy), 0.5)) / (pow(ky / kx, 0.25) + pow(kx / ky, 0.25));
#     WI_ = alpha * 2 * PI*pow(kx*ky, 0.5)*dz / log( r0_ / r_);
    r0 = 0.28 * np.sqrt(dx*dx + dy*dy) / 2.
    wi = alpha * 2. * np.pi * k * dz / np.log(r0 / r)
    wi = 14 / 5
    return wi


def cal_kr(sw):
    swi, sor, aw, ao, krw0, kro0 = 0.1, 0.3, 1.5, 3.6, 0.7, 1.0
    ds = 1. - swi - sor
    if sw < swi:
        krw = 0.0
    elif sw>=1-sor:
        krw = krw0
    else:
        sw_norm = (sw - swi) / ds
        krw = krw0 * (sw_norm ** aw)
    so = 1 - sw
    if so < sor:
        kro = 0.0
    elif so >= 1 - swi:
        kro = kro0
    else:
        so_norm = (so - sor) / ds
        kro = kro0 * (so_norm ** ao)
    return krw, kro


def cal_pvto(p, pvdo_table):
    p_table = pvdo_table[:, 0]
    bo_table = pvdo_table[:, 1]
    vo_table = pvdo_table[:, 2]
    bo = np.interp(p, p_table, bo_table)
    vo = np.interp(p, p_table, vo_table)
    return bo, vo


def cal_pvtw(p):
    pw_ref, bw_ref, cw, vw_ref, cv = 14.5038, 1.00, 4e-5, 0.31, 0.0
    y = -cv * (p - pw_ref)
    vw = vw_ref / (1. + y + y*y/2.0)
    x = cw * (p - pw_ref)
    bw = bw_ref / (1. + x + x*x/2.0)
    return bw, vw


def cal_mobi(p, sw):
    krw, kro = cal_kr(sw)
    pvdo_table = load_pvdo()
    bo, vo = cal_pvto(p, pvdo_table)
    bw, vw = cal_pvtw(p)
    mo = kro/(vo*bo)
    mw = krw/(vw*bw)
    return mw, mo


def cal_prod_rate(p, sw, bhp, k):
    wi = cal_well_index(k)

    nt = len(p)
    orat = np.zeros((nt, ))
    wrat = np.zeros((nt, ))

    for i in range(nt):
        pt = p[i]
        swt = sw[i]

        mw, mo = cal_mobi(pt, swt)
        orat[i] = wi * mo * (pt - bhp)
        wrat[i] = wi * mw * (pt - bhp)
    return wrat, orat


def cal_inj_bhp(p, sw, wrat, k):
    wi = cal_well_index(k)

    nt = len(p)
    bhp = np.zeros((nt,))
    for i in range(nt):
        pt = p[i]
        swt = sw[i]

        mw, mo = cal_mobi(pt, swt)
        bhp[i] = p - wrat / (wi*(mo + mw))
    
    return bhp


