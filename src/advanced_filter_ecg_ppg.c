#include "advanced_filter_ecg_ppg.h"
#include "logger.h"

typedef union _tag_caster {
   uint32_t u32;
   uint16_t u16[2];
   uint8_t u8[4];
} caster;

static uint16_t ecg[WINDOW];
static uint16_t ppg[WINDOW];
static uint8_t ecg_parsed[WINDOW];
static uint8_t ppg_parsed[WINDOW];
static uint32_t mark_ecg[DOTS];
static uint32_t mark_ppg[DOTS];

static void init_adv_filter_input(adv_filter* filt, uint16_t data_ecg, uint16_t data_ppg);
static void after_adv_filter_input(adv_filter* filt, uint16_t data_ecg, uint16_t data_ppg);
static inline void adv_min_max(adv_filter* filt, uint16_t data_ecg, uint16_t data_ppg);
static inline void adv_min_max_clear(adv_filter* filt);
static inline void adv_min_max_stage(adv_filter* filt);
static inline void adv_correction(adv_filter* filt, const uint16_t data_ecg, const uint16_t data_ppg);

void adv_filter_init(adv_filter* filt)
{
    TRACE_LOG("adv_filter_init\n");
    filt->ecg = ecg;
    filt->ppg = ppg;
    filt->ecg_parsed = ecg_parsed;
    filt->ppg_parsed = ppg_parsed;
    filt->mark_ecg = mark_ecg;
    filt->mark_ppg = mark_ppg;
    filt->window = WINDOW;
    memset(filt->ecg, 0, sizeof(ecg));
    memset(filt->ppg, 0, sizeof(ppg));
    memset(filt->ecg_parsed, 0, sizeof(ecg_parsed));
    memset(filt->ppg_parsed, 0, sizeof(ppg_parsed));
    memset(filt->mark_ecg, 0, sizeof(mark_ecg));
    memset(filt->mark_ppg, 0, sizeof(mark_ppg));
    adv_min_max_clear(filt);
    filt->tail = 0;
    filt->head = 0;
    filt->head_parsed = 0;
    filt->mark_head = 0;
    filt->fp = &init_adv_filter_input;
    TRACE_LOG("adv_filter_init end\n");
}

static inline void adv_min_max_clear(adv_filter* filt)
{
    TRACE_LOG("adv_min_max_clear\n");
    filt->min_ecg[0] = UINT_LEAST16_MAX;
    filt->max_ecg[0] = 0;
    filt->min_ecg_it[0] = 0;
    filt->max_ecg_it[0] = 0;
    filt->min_ppg[0] = UINT_LEAST16_MAX;
    filt->max_ppg[0] = 0;
    filt->min_ppg_it[0] = 0;
    filt->max_ppg_it[0] = 0;
    TRACE_LOG("adv_min_max_clear end\n");
}

static void init_adv_filter_input(adv_filter* filt, uint16_t data_ecg, uint16_t data_ppg)
{
    TRACE_LOG("init_adv_filter_input\n");
    ecg[filt->head % WINDOW] = data_ecg;
    ppg[filt->head % WINDOW] = data_ppg;
    filt->head++;
    TRACE_LOG("head = %zu\n", filt->head);
    if (filt->head >= WINDOW)
    {
        DEBUG_LOG("Start window collected\n");
        filt->fp = &after_adv_filter_input;
        TRACE_LOG("init_adv_filter_input end\n");
        return;
    }
    TRACE_LOG("init_adv_filter_input end\n");
}

static void after_adv_filter_input(adv_filter* filt, uint16_t data_ecg, uint16_t data_ppg)
{
    TRACE_LOG("after_adv_filter_input\n");
    size_t i;
    const uint16_t old_ecg = ecg[filt->head % WINDOW];
    const uint16_t old_ppg = ppg[filt->head % WINDOW];
    ecg[filt->head % WINDOW] = data_ecg;
    ppg[filt->head % WINDOW] = data_ppg;
    filt->tail++;
    filt->head++;
    if ( (filt->min_ecg_it < filt->tail)
            || (filt->max_ecg_it < filt->tail)
            || (filt->min_ppg_it < filt->tail)
            || (filt->max_ppg_it < filt->tail) )
    {
        DEBUG_LOG("One of min/max out of range\n");
        adv_min_max_clear(filt);
        for (i = 0;i < WINDOW;i++)
        {
            DEBUG_LOG("Finding new one %zu\n", i);
            adv_min_max(filt, ecg[i], ppg[i]);
        }
    }
    adv_correction(filt, old_ecg, old_ppg);
    TRACE_LOG("after_adv_filter_input end\n");
}

static inline void adv_correction(adv_filter* filt, const uint16_t data_ecg, const uint16_t data_ppg)
{
    TRACE_LOG("adv_correction\n");
    uint32_t ecg_t = data_ecg - filt->min_ecg[2];
    uint32_t ppg_t = data_ppg - filt->min_ecg[2];
    PRECISE_LOG("selected %u = %hu - %hu\n", ecg_t, data_ecg, filt->min_ecg[2]);
    PRECISE_LOG("selected %u = %hu - %hu\n", ppg_t, data_ppg, filt->min_ppg[2]);
    ecg_t = ecg_t * UINT_LEAST8_MAX / (uint32_t)(filt->max_ecg[2] - filt->min_ecg[2]);
    ppg_t = ppg_t * UINT_LEAST8_MAX / (uint32_t)(filt->max_ppg[2] - filt->min_ppg[2]);
    PRECISE_LOG("resized %u *= 1 / (%hu - %hu)\n",
            ecg_t, filt->max_ecg[2], filt->min_ecg[2]);
    PRECISE_LOG("resized %u *= 1 / (%hu - %hu)\n",
            ppg_t, filt->max_ppg[2], filt->min_ppg[2]);
    ecg_parsed[filt->head_parsed % WINDOW] = (uint8_t)ecg_t;
    ppg_parsed[filt->head_parsed % WINDOW] = (uint8_t)ppg_t;
    PRECISE_LOG("saved %hhu (%zu)\n", ecg_t, filt->head_parsed % WINDOW);
    PRECISE_LOG("saved %hhu (%zu)\n", ppg_t, filt->head_parsed % WINDOW);
    filt->head_parsed++;
    TRACE_LOG("adv_correction end\n");
}

void adv_filter_input(adv_filter* filt, uint16_t data_ecg, uint16_t data_ppg)
{
    TRACE_LOG("adv_filter_input (%hu, %hu)\n",data_ecg,data_ppg);
    data_ecg++;
    data_ppg++;
    (filt->fp)(filt, data_ecg, data_ppg);
    adv_min_max(filt, data_ecg, data_ppg);
    if (!(filt->head % (WINDOW / 2)))
    {
        filt->mark_ecg[filt->mark_head % DOTS] = filt->max_ecg_it[2];
        filt->mark_head++;
        adv_min_max_stage(filt);
        filt->min_ecg[1] = filt->min_ecg[0];
        filt->max_ecg[1] = filt->max_ecg[0];
        filt->min_ecg_it[1] = filt->min_ecg_it[0];
        filt->max_ecg_it[1] = filt->max_ecg_it[0];
        filt->min_ppg[1] = filt->min_ppg[0];
        filt->max_ppg[1] = filt->max_ppg[0];
        filt->min_ppg_it[1] = filt->min_ppg_it[0];
        filt->max_ppg_it[1] = filt->max_ppg_it[0];
    }
    TRACE_LOG("adv_filter_input end\n");
}

void adv_filter_inputs(adv_filter* filt, uint16_t* data_ecg, uint16_t* data_ppg, size_t n)
{
    TRACE_LOG("adv_filter_inputs\n");
    size_t i;
    for (i = 0;i < n;i++)
    {
        DEBUG_LOG("Pass data to filter %zu\n", i);
        adv_filter_input(filt, data_ecg[i], data_ppg[i]);
    }
    TRACE_LOG("adv_filter_inputs end\n");
}

#ifdef SIMD
//res[15:0]  = val1[15:0]  - val2[15:0]
//res[31:16] = val1[31:16] - val2[31:16]
//__SSUB16(cast1, cast2);
//if res[15:0] >= 0 then APSR.GE[1:0] = 11 else 00
//if res[31:16] >= 0 then APSR.GE[3:2] = 11 else 00
//if APSR.GE[0] == 1 then res[7:0] = val1[7:0] else res[7:0] = val2[7:0]
//if APSR.GE[1] == 1 then res[15:8] = val1[15:8] else res[15:8] = val2[15:8]
//if APSR.GE[2] == 1 then res[23:16] = val1[23:16] else res[23:16] = val2[23:16]
//if APSR.GE[3] == 1 then res[31;24] = val1[31:24] else res = val2[31:24]
//res = __SEL(cast1, cast2);
static inline void adv_min_max(adv_filter* filt, uint16_t data_ecg, uint16_t data_ppg)
{
    TRACE_LOG("adv_min_max\n");
    caster cast1, cast2, res;
    cast1.u16[0] = filt->min_ecg[0];
    cast1.u16[1] = filt->min_ppg[0];
    cast2.u16[0] = data_ecg;
    cast2.u16[1] = data_ppg;
    PRECISE_LOG("cast1(%hu, %hu) cast2(%hu, %hu)\n",
            filt->min_ecg[0], filt->min_ppg[0], data_ecg, data_ppg);
    __SSUB16(cast1, cast2);
    res = __SEL(cast1, cast2);
    PRECISE_LOG("__SSUB16 & __SEL\n");
    filt->min_ecg[0] = res.u16[0];
    filt->min_ppg[0] = res.u16[1];
    PRECISE_LOG("result(%hu, %hu)\n", filt->min_ecg[0], filt->min_ppg[0]);
    cast1.u16[0] = filt->max_ecg[0];
    cast1.u16[1] = filt->max_ppg[0];
    PRECISE_LOG("cast1(%hu, %hu) cast2(%hu, %hu)\n",
            filt->max_ecg[0], filt->max_ppg[0], data_ecg, data_ppg);
    __SSUB16(cast1, cast2);
    res = __SEL(cast2, cast1);
    PRECISE_LOG("__SSUB16 & __SEL\n");
    filt->max_ecg[0] = res.u16[0];
    filt->max_ppg[0] = res.u16[1];
    PRECISE_LOG("result(%hu, %hu)\n", filt->max_ecg[0], filt->max_ppg[0]);
    TRACE_LOG("adv_min_max end\n");
}
static inline void adv_min_max_stage(adv_filter* filt)
{
    TRACE_LOG("adv_min_max\n");
    caster cast1, cast2, res;
    cast1.u16[0] = filt->min_ecg[1];
    cast1.u16[1] = filt->min_ppg[1];
    cast2.u16[0] = filt->min_ecg[0];
    cast2.u16[1] = filt->min_ppg[0];
    PRECISE_LOG("cast1(%hu, %hu) cast2(%hu, %hu)\n",
            filt->min_ecg[1], filt->min_ppg[1], filt->min_ecg[0], filt->min_ppg[0]);
    __SSUB16(cast1, cast2);
    res = __SEL(cast1, cast2);
    PRECISE_LOG("__SSUB16 & __SEL\n");
    filt->min_ecg[2] = res.u16[0];
    filt->min_ppg[2] = res.u16[1];
    PRECISE_LOG("result(%hu, %hu)\n", filt->min_ecg[2], filt->min_ppg[2]);
    cast1.u16[0] = filt->max_ecg[1];
    cast1.u16[1] = filt->max_ppg[1];
    cast2.u16[0] = filt->max_ecg[0];
    cast2.u16[1] = filt->max_ppg[0];
    PRECISE_LOG("cast1(%hu, %hu) cast2(%hu, %hu)\n",
            filt->max_ecg[1], filt->max_ppg[1], filt->max_ecg[0], filt->max_ppg[0]);
    __SSUB16(cast1, cast2);
    res = __SEL(cast2, cast1);
    PRECISE_LOG("__SSUB16 & __SEL\n");
    filt->max_ecg[2] = res.u16[0];
    filt->max_ppg[2] = res.u16[1];
    PRECISE_LOG("result(%hu, %hu)\n", filt->max_ecg[2], filt->max_ppg[2]);
    TRACE_LOG("adv_min_max end\n");
}
#else
//(result && (max = a)) || (max = b)
static inline void adv_min_max(adv_filter* filt, uint16_t data_ecg, uint16_t data_ppg)
{
    TRACE_LOG("adv_min_max\n");
    const uint16_t min_ecg_cmp = filt->min_ecg[0] < data_ecg;
    const uint16_t min_ppg_cmp = filt->min_ppg[0] < data_ppg;
    const uint16_t max_ecg_cmp = filt->max_ecg[0] > data_ecg;
    const uint16_t max_ppg_cmp = filt->max_ppg[0] > data_ppg;
    
    PRECISE_LOG("min_ecg(%hu, %hu)", filt->min_ecg[0], data_ecg);
    (min_ecg_cmp && (filt->min_ecg[0] = filt->min_ecg[0]))
            || (filt->min_ecg[0] = data_ecg);
    PRECISE_LOG(" = %hu ", filt->min_ecg[0]);
    (min_ecg_cmp && (filt->min_ecg_it[0] = filt->min_ecg_it[0]))
            || (filt->min_ecg_it[0] = (filt->head - 1));
    PRECISE_LOG("(%zu)\n", filt->min_ecg_it[0]);
    
    PRECISE_LOG("min_ppg(%hu, %hu)", filt->min_ppg[0], data_ppg);
    (min_ppg_cmp && (filt->min_ppg[0] = filt->min_ppg[0]))
            || (filt->min_ppg[0] = (data_ppg));
    PRECISE_LOG(" = %hu ", filt->min_ppg[0]);
    (min_ppg_cmp && (filt->min_ppg_it[0] = filt->min_ppg_it[0]))
            || (filt->min_ppg_it[0] = (filt->head - 1));
    PRECISE_LOG("(%zu)\n", filt->min_ppg_it[0]);
    
    PRECISE_LOG("max_ecg(%hu, %hu)", filt->max_ecg[0], data_ecg);
    (max_ecg_cmp && (filt->max_ecg[0] = filt->max_ecg[0]))
            || (filt->max_ecg[0] = data_ecg);
    PRECISE_LOG(" = %hu ", filt->max_ecg[0]);
    (max_ecg_cmp && (filt->max_ecg_it[0] = filt->max_ecg_it[0]))
            || (filt->max_ecg_it[0] = (filt->head - 1));
    PRECISE_LOG("(%zu)\n", filt->max_ecg_it[0]);
    
    PRECISE_LOG("max_ppg(%hu, %hu)", filt->max_ppg[0], data_ppg);
    (max_ppg_cmp && (filt->max_ppg[0] = filt->max_ppg[0]))
            || (filt->max_ppg[0] = (data_ppg));
    PRECISE_LOG(" = %hu ", filt->max_ppg[0]);
    (max_ppg_cmp && (filt->max_ppg_it[0] = filt->max_ppg_it[0]))
            || (filt->max_ppg_it[0] = (filt->head - 1));
    PRECISE_LOG("(%zu)\n", filt->max_ppg_it[0]);
    TRACE_LOG("adv_min_max end\n");
}
static inline void adv_min_max_stage(adv_filter* filt)
{
    TRACE_LOG("adv_min_max_stage\n");
    const uint16_t min_ecg_cmp = filt->min_ecg[1] < filt->min_ecg[0];
    const uint16_t min_ppg_cmp = filt->min_ppg[1] < filt->min_ppg[0];
    const uint16_t max_ecg_cmp = filt->max_ecg[1] > filt->max_ecg[0];
    const uint16_t max_ppg_cmp = filt->max_ppg[1] > filt->max_ppg[0];
    
    PRECISE_LOG("min_ecg(%hu, %hu)", filt->min_ecg[1], filt->min_ecg[0]);
    (min_ecg_cmp && (filt->min_ecg[2] = filt->min_ecg[1]))
            || (filt->min_ecg[2] = filt->min_ecg[0]);
    PRECISE_LOG(" = %hu ", filt->min_ecg[2]);
    (min_ecg_cmp && (filt->min_ecg_it[2] = filt->min_ecg_it[1]))
            || (filt->min_ecg_it[2] = filt->min_ecg_it[0]);
    PRECISE_LOG("(%zu)\n", filt->min_ecg_it[2]);
    
    PRECISE_LOG("min_ppg(%hu, %hu)", filt->min_ppg[1], filt->min_ppg[0]);
    (min_ppg_cmp && (filt->min_ppg[2] = filt->min_ppg[1]))
            || (filt->min_ppg[2] = filt->min_ppg[0]);
    PRECISE_LOG(" = %hu ", filt->min_ppg[2]);
    (min_ppg_cmp && (filt->min_ppg_it[2] = filt->min_ppg_it[1]))
            || (filt->min_ppg_it[2] = filt->min_ppg_it[0]);
    PRECISE_LOG("(%zu)\n", filt->min_ppg_it[2]);
    
    PRECISE_LOG("max_ecg(%hu, %hu)", filt->max_ecg[1], filt->max_ecg[0]);
    (max_ecg_cmp && (filt->max_ecg[2] = filt->max_ecg[1]))
            || (filt->max_ecg[2] = filt->max_ecg[0]);
    PRECISE_LOG(" = %hu ", filt->max_ecg[2]);
    (max_ecg_cmp && (filt->max_ecg_it[2] = filt->max_ecg_it[1]))
            || (filt->max_ecg_it[2] = filt->max_ecg_it[0]);
    PRECISE_LOG("(%zu)\n", filt->max_ecg_it[2]);
    
    PRECISE_LOG("max_ppg(%hu, %hu)", filt->max_ppg[1], filt->max_ppg[0]);
    (max_ppg_cmp && (filt->max_ppg[2] = filt->max_ppg[1]))
            || (filt->max_ppg[2] = filt->max_ppg[0]);
    PRECISE_LOG(" = %hu ", filt->max_ppg[2]);
    (max_ppg_cmp && (filt->max_ppg_it[2] = filt->max_ppg_it[1]))
            || (filt->max_ppg_it[2] = filt->max_ppg_it[0]);
    PRECISE_LOG("(%zu)\n", filt->max_ppg_it[2]);
    TRACE_LOG("adv_min_max_stage end\n");
}
#endif