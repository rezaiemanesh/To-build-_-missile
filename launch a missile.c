/* 6-State 3-Axis Kalman Filter for IMU Attitude Estimation
   Author: Mohamad Ali Rezaiemanesh (rezaiemanesh@chmail.ir) – July 2020 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#define DEG2RAD (0.017453292519943295f)

/* ---------- Linear-algebra helpers (minimal) ---------- */
static void mat6_add(float A[6][6], const float B[6][6]) {
    for (int i=0;i<6;i++) for (int j=0;j<6;j++) A[i][j]+=B[i][j];
}
static void mat6_sub(float A[6][6], const float B[6][6]) {
    for (int i=0;i<6;i++) for (int j=0;j<6;j++) A[i][j]-=B[i][j];
}
static void mat6_mult(float C[6][6], const float A[6][6], const float B[6][6]) {
    float tmp[6][6]={0};
    for(int i=0;i<6;i++)
        for(int k=0;k<6;k++)
            for(int j=0;j<6;j++)
                tmp[i][j]+=A[i][k]*B[k][j];
    memcpy(C,tmp,sizeof(tmp));
}
static void mat6_transpose(float T[6][6], const float A[6][6]) {
    for(int i=0;i<6;i++) for(int j=0;j<6;j++) T[i][j]=A[j][i];
}

/* ---------- Kalman structure ---------- */
typedef struct {
    float x[6];        /* state: [roll pitch yaw b_roll b_pitch b_yaw] */
    float P[6][6];     /* covariance */
    float Q[6][6];     /* process noise */
    float R[3][3];     /* measurement noise (3 angles) */
} Kalman6D;

void kalman6_init(Kalman6D *kf, float q_angle, float q_bias, float r_measure) {
    memset(kf,0,sizeof(*kf));

    /* initialize covariances */
    for(int i=0;i<3;i++) kf->P[i][i]=1.0f;        /* angle var */
    for(int i=3;i<6;i++) kf->P[i][i]=1.0f;        /* bias var */

    /* process noise */
    for(int i=0;i<3;i++) kf->Q[i][i]=q_angle;
    for(int i=3;i<6;i++) kf->Q[i][i]=q_bias;

    /* measurement noise (acc+mag angles) */
    for(int i=0;i<3;i++) kf->R[i][i]=r_measure;
}

/* helper to build state-transition matrix F = I + dt * A */
static void build_F(float F[6][6], float dt) {
    memset(F,0,sizeof(float)*36);
    for(int i=0;i<6;i++) F[i][i]=1.0f;
    /* d(angle)/dt = (gyro - bias) -> angle depends on bias term with -dt */
    F[0][3]=-dt;
    F[1][4]=-dt;
    F[2][5]=-dt;
}

/* Time-update (prediction) */
void kalman6_predict(Kalman6D *kf, const float gyro[3], float dt) {
    /* state propagation x = x + dt*(gyro-bias) */
    for(int i=0;i<3;i++) {
        float rate = (gyro[i]*DEG2RAD) - kf->x[3+i];
        kf->x[i] += dt*rate;
    }
    /* build F and update P = F P F^T + Q */
    float F[6][6], Ft[6][6], FP[6][6];
    build_F(F,dt);
    mat6_mult(FP,F,kf->P);
    mat6_transpose(Ft,F);
    mat6_mult(kf->P,FP,Ft);
    mat6_add(kf->P,kf->Q);
}

/* Compute roll & pitch from accelerometer, yaw from magnetometer */
static void acc_mag_to_angles(float *angles, const float acc[3], const float mag[3]) {
    /* roll */
    angles[0] = atan2f(acc[1],acc[2]);
    /* pitch */
    angles[1] = -atan2f(acc[0], sqrtf(acc[1]*acc[1]+acc[2]*acc[2]));
    /* yaw (simple tilt-compensated) */
    float mx=mag[0]*cosf(angles[1]) + mag[2]*sinf(angles[1]);
    float my=mag[0]*sinf(angles[0])*sinf(angles[1]) + mag[1]*cosf(angles[0]) - mag[2]*sinf(angles[0])*cosf(angles[1]);
    angles[2] = atan2f(-my,mx);
}

/* Measurement-update (correction) */
void kalman6_update(Kalman6D *kf, const float acc[3], const float mag[3]) {
    float z[3];
    acc_mag_to_angles(z,acc,mag);     /* measured angles */

    /* innovation y = z - Hx   (H extracts first 3 states) */
    float y[3];
    for(int i=0;i<3;i++) y[i]=z[i]-kf->x[i];

    /* S = H P H^T + R   -> since H=[I3 03×3], extract upper-left 3×3 block */
    float S[3][3];
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            S[i][j]=kf->P[i][j] + kf->R[i][j];

    /* compute Kalman gain K = P H^T S^{-1}; here K is 6×3 */
    /* invert 3×3 S (assuming symmetric + positive-def) */
    float det = S[0][0]*(S[1][1]*S[2][2]-S[1][2]*S[2][1])
              - S[0][1]*(S[1][0]*S[2][2]-S[1][2]*S[2][0])
              + S[0][2]*(S[1][0]*S[2][1]-S[1][1]*S[2][0]);
    if (fabsf(det)<1e-9f) return; /* singular, skip update */

    float invS[3][3];
    invS[0][0]=(S[1][1]*S[2][2]-S[1][2]*S[2][1])/det;
    invS[0][1]=(S[0][2]*S[2][1]-S[0][1]*S[2][2])/det;
    invS[0][2]=(S[0][1]*S[1][2]-S[0][2]*S[1][1])/det;
    invS[1][0]=(S[1][2]*S[2][0]-S[1][0]*S[2][2])/det;
    invS[1][1]=(S[0][0]*S[2][2]-S[0][2]*S[2][0])/det;
    invS[1][2]=(S[0][2]*S[1][0]-S[0][0]*S[1][2])/det;
    invS[2][0]=(S[1][0]*S[2][1]-S[1][1]*S[2][0])/det;
    invS[2][1]=(S[0][1]*S[2][0]-S[0][0]*S[2][1])/det;
    invS[2][2]=(S[0][0]*S[1][1]-S[0][1]*S[1][0])/det;

    /* K = P(:,0:2) * invS */
    float K[6][3]={0};
    for(int i=0;i<6;i++)
        for(int j=0;j<3;j++)
            for(int k=0;k<3;k++)
                K[i][j]+=kf->P[i][k]*invS[k][j];

    /* state update x = x + K*y */
    for(int i=0;i<6;i++)
        for(int j=0;j<3;j++)
            kf->x[i]+=K[i][j]*y[j];

    /* covariance update P = (I - K H) P   (H selects first 3 rows)   */
    float KH[6][6]={0};
    for(int i=0;i<6;i++)
        for(int j=0;j<3;j++)
            KH[i][j]=K[i][j];      /* only first 3 cols non-zero */

    float I_KH[6][6];
    for(int i=0;i<6;i++)
        for(int j=0;j<6;j++)
            I_KH[i][j]= (i==j) - KH[i][j];

    float tmp[6][6];
    mat6_mult(tmp,I_KH,kf->P);
    memcpy(kf->P,tmp,sizeof(tmp));
}

/* ---------- Example usage ---------- */
int main(void) {
    Kalman6D kf;
    kalman6_init(&kf, 0.0005f, 0.0001f, 0.02f);

    float dt=0.01f;  /* 100 Hz loop */

    /* Here, replace with real IMU streaming loop */
    for(int step=0; step<1000; ++step) {
        float gyro[3] = {0.3f, 0.0f, -0.2f};  /* °/s */
        float acc[3]  = {0.0f, 0.0f, 1.0f};   /* g */
        float mag[3]  = {0.2f, 0.0f, 0.4f};   /* µT */

        kalman6_predict(&kf, gyro, dt);
        kalman6_update(&kf, acc, mag);

        printf("t=%.2fs  Roll=%.2f°  Pitch=%.2f°  Yaw=%.2f°\n",
               step*dt, kf.x[0]*57.29578f, kf.x[1]*57.29578f, kf.x[2]*57.29578f);
    }
    return 0;
}
