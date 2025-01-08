#pragma once

#include <random>
#include <array>
#include <cstring>
#define ITER 512

/*
 * ================ DO NOT MODIFY ================
 * =========== SHA256 MACROS AND TYPES ===========
 */
using BYTE = unsigned char;
using WORD = uint32_t;

constexpr WORD ROTLEFT(WORD a, WORD b) { return (a << b) | (a >> (32 - b)); }

constexpr WORD ROTRIGHT(WORD a, WORD b) { return (a >> b) | (a << (32 - b)); }

constexpr WORD CH(WORD x, WORD y, WORD z) { return (x & y) ^ (~x & z); }

constexpr WORD MAJ(WORD x, WORD y, WORD z) { return (x & y) ^ (x & z) ^ (y & z); }

constexpr WORD EP0(WORD x) { return ROTRIGHT(x, 2) ^ ROTRIGHT(x, 13) ^ ROTRIGHT(x, 22); }

constexpr WORD EP1(WORD x) { return ROTRIGHT(x, 6) ^ ROTRIGHT(x, 11) ^ ROTRIGHT(x, 25); }

constexpr WORD SIG0(WORD x) { return ROTRIGHT(x, 7) ^ ROTRIGHT(x, 18) ^ (x >> 3); }

constexpr WORD SIG1(WORD x) { return ROTRIGHT(x, 17) ^ ROTRIGHT(x, 19) ^ (x >> 10); }

constexpr std::array<WORD, 64> k{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

struct SHA256_CTX {
    std::array<BYTE, 64> data;
    WORD datalen;
    unsigned long long bitlen;
    std::array<WORD, 8> state;
};

void sha256_transform(SHA256_CTX &ctx, const std::array<BYTE, 64> &data) {
    WORD a, b, c, d, e, f, g, h, i, j, t1, t2;
    std::array<WORD, 64> m;

    for (i = 0, j = 0; i < 16; ++i, j += 4) {
        m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
    }

    for (; i < 64; ++i) {
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
    }

    a = ctx.state[0];
    b = ctx.state[1];
    c = ctx.state[2];
    d = ctx.state[3];
    e = ctx.state[4];
    f = ctx.state[5];
    g = ctx.state[6];
    h = ctx.state[7];

    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e, f, g) + k[i] + m[i];
        t2 = EP0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    ctx.state[0] += a;
    ctx.state[1] += b;
    ctx.state[2] += c;
    ctx.state[3] += d;
    ctx.state[4] += e;
    ctx.state[5] += f;
    ctx.state[6] += g;
    ctx.state[7] += h;
}

void sha256_init(SHA256_CTX &ctx) {
    ctx.datalen = 0;
    ctx.bitlen = 0;
    ctx.state[0] = 0x6a09e667;
    ctx.state[1] = 0xbb67ae85;
    ctx.state[2] = 0x3c6ef372;
    ctx.state[3] = 0xa54ff53a;
    ctx.state[4] = 0x510e527f;
    ctx.state[5] = 0x9b05688c;
    ctx.state[6] = 0x1f83d9ab;
    ctx.state[7] = 0x5be0cd19;
}

void sha256_update(SHA256_CTX &ctx, const BYTE *data, size_t len) {
    WORD i;

    for (i = 0; i < len; ++i) {
        ctx.data[ctx.datalen] = data[i];
        ctx.datalen++;
        if (ctx.datalen == 64) {
            sha256_transform(ctx, ctx.data);
            ctx.bitlen += 512;
            ctx.datalen = 0;
        }
    }
}

void sha256_final(SHA256_CTX &ctx, std::array<BYTE, 32> &hash) {
    WORD i = ctx.datalen;

    // Pad whatever data is left in the buffer
    if (ctx.datalen < 56) {
        ctx.data[i++] = 0x80;
        while (i < 56) {
            ctx.data[i++] = 0x00;
        }
    } else {
        ctx.data[i++] = 0x80;
        while (i < 64) {
            ctx.data[i++] = 0x00;
        }
        sha256_transform(ctx, ctx.data);
        memset(ctx.data.data(), 0, 56);
    }

    // Append to the padding the total message's length in bits and transform
    ctx.bitlen += ctx.datalen * 8;
    ctx.data[63] = ctx.bitlen;
    ctx.data[62] = ctx.bitlen >> 8;
    ctx.data[61] = ctx.bitlen >> 16;
    ctx.data[60] = ctx.bitlen >> 24;
    ctx.data[59] = ctx.bitlen >> 32;
    ctx.data[58] = ctx.bitlen >> 40;
    ctx.data[57] = ctx.bitlen >> 48;
    ctx.data[56] = ctx.bitlen >> 56;
    sha256_transform(ctx, ctx.data);

    // Since this implementation uses little endian byte ordering and SHA uses
    // big endian, reverse all the bytes when copying the final state to the
    // output hash
    for (i = 0; i < 4; ++i) {
        hash[i] = (ctx.state[0] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 4] = (ctx.state[1] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 8] = (ctx.state[2] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 12] = (ctx.state[3] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 16] = (ctx.state[4] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 20] = (ctx.state[5] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 24] = (ctx.state[6] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 28] = (ctx.state[7] >> (24 - i * 8)) & 0x000000ff;
    }
}

// ============= END SHA-256 SECTION =============

class PlatformLoadTimeGen {
  private:
    int popularity;
    std::mt19937_64 gen;
    uint64_t last_value;

    void reseed(uint64_t entropy) {
        SHA256_CTX ctx;
        std::array<BYTE, 32> hash;
        // Linear congruential generator XDD
        uint64_t combined_seed = entropy ^ (last_value * 6364136223846793005ULL);

        // artificial workload for generating deterministic loading times, do not change!
        for (short i = 0; i < ITER; i++) {
            combined_seed = combined_seed + (last_value >> 32);
            std::array<BYTE, sizeof(combined_seed)> byte_arr;
            std::memcpy(byte_arr.data(), &combined_seed, sizeof(combined_seed));

            sha256_init(ctx);
            sha256_update(ctx, byte_arr.data(), sizeof(byte_arr));
            sha256_final(ctx, hash);

            for (short j = 0; j < 32; j += 8) {
                uint64_t *hash_part = (uint64_t *)(&hash[j]);
                combined_seed ^= *hash_part;
            }
        }

        gen.seed(combined_seed);
        last_value = combined_seed;
    }

  public:
    PlatformLoadTimeGen(int popularity) : popularity(popularity), gen(3210), last_value(3210) {}

    int next(int train_id) {
        // min waiting time & popularity must be 1
        std::poisson_distribution<int> dist(popularity - 1);
        int next_waiting_time = dist(gen) + 1;
        reseed(train_id);
        return next_waiting_time;
    }
};
