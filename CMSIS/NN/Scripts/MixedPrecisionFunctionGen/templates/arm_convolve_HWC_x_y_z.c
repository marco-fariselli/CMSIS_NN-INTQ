/*
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.
 * Modifications Copyright (C) 2018 University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library - Mixed Precision INT-Q
 * Title:        ${config.filename}
 * Description:  Mixed Precision Convolutional function that uses ${config.in_data_t}
 *               activations, ${config.wt_data_t} weights and produce ${config.out_data_t}
 *               output activations. Outputs are quantized using ${config.folding}
 *               folding technique.
 *
 * $Date:        March 2019
 * $Authors:     Alessandro Capotondi - alessandro.capotondi@unibo.it
 *               Manuele Rusci - manuele.rusci@unibo.it
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */
#include <assert.h>

#include "arm_math.h"
#include "arm_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

/**
   * @brief Mixed Precision Convolution ${config.folding} (in: ${config.in_data_t}, out: ${config.out_data_t}, wt: ${config.wt_data_t})
   *
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       left_pad    padding sizes
   * @param[in]       right_pad   padding sizes
   * @param[in]       top_pad     padding sizes
   * @param[in]       bottom_pad  padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in]       z_in        input offset
% if config.quantization=="PACT":
   * @param[in]       z_wt        weights offset
% elif config.quantization=="PACT_CH":
   * @param[in]       *z_wt       weights offset, per-output channel
% endif
% if config.folding=="thr":
   * @param[in]       thresholds  pointer to thresholds
% elif config.folding=="icn":
   * @param[in]       z_out       output offset
   * @param[in]       *m_zero     pointer to m zero quantization params (per-output-ch)
   * @param[in]       *n_zero     pointer to n zero quantization params (per-output-ch)
% else:
   * @param[in]       z_out       output offset
   * @param[in]       m_zero      m zero quantization param
   * @param[in]       n_zero      n zero quantization param
% endif
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   */
arm_status
${config.fn_name}(const uint8_t *Im_in,
                    const uint16_t dim_im_in,
                    const uint16_t ch_im_in,
                    const uint8_t *wt,
                    const uint16_t ch_im_out,
                    const uint16_t dim_kernel,
                    const uint8_t left_padding,
                    const uint8_t right_padding,
                    const uint8_t top_padding,
                    const uint8_t bottom_padding,
                    const uint16_t stride,
                    const int32_t *bias,
                    uint8_t *Im_out,
                    const uint16_t dim_im_out,
                    const uint8_t z_in,
% if config.quantization == "PACT":
                    const uint8_t z_wt,
% elif config.quantization == "PACT_CH":
                    const uint8_t *z_wt,
% endif
% if config.folding == "thr":
                    const int16_t *thresholds,
% else:
                    const uint8_t z_out,
%   if config.folding == "icn":
                    const int32_t *m_zero,
                    const int8_t *n_zero,
%   else:
                    const int32_t m_zero,
                    const int8_t n_zero,
%   endif
% endif
                    int16_t * bufferA,
                    uint8_t *bufferB)
{

#if defined(ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t i_out_y, i_out_x, i_ker_y, i_ker_x;
    int16_t *pBuffer = bufferA;
    uint8_t *pOut = Im_out;

    if (ch_im_in % ${config.ch_in_constrain} != 0 || ch_im_out % ${config.ch_out_constrain} != 0)
    {
        /* check if the input dimension meets the constraints */
        return ARM_MATH_SIZE_MISMATCH;
    }

    /*
     *  Here we split the entire matrix into three regions depending on the padding situation
     *    Top: i_out_y from 0 to padding - 1
     * Middle: i_out_y from padding to dim_im_out-padding-1
     * Bottom: i_out_y from dim_im_out-padding to dim_im_out-1
     */

    /* top part */
    for (i_out_y = 0; i_out_y < top_padding; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - top_padding; i_ker_y < i_out_y * stride - top_padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - left_padding; i_ker_x < i_out_x * stride - left_padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        memset(pBuffer, 0, sizeof(int16_t) * ch_im_in);
                    }
                    else
                    {
                        ${config.reordered_no_shift_load_fn}(
% if config.in_data_t == 'u8':
                                                Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in,
% elif config.in_data_t == 'u4':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1),
% elif config.in_data_t == 'u2':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 2),
% endif
                                                pBuffer,
                                                ch_im_in,
                                                z_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut = ${config.nn_mat_mul_fn}(wt,
                                                bufferA,
                                                ch_im_out,
                                                ch_im_in * dim_kernel * dim_kernel,
                                                bias,
                                                pOut,
                                                z_wt,
% if config.folding == "thr":
                                                thresholds);
% else:
                                                z_out,
                                                m_zero,
                                                n_zero);
% endif
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    /* middle part, here we also divide the x into left, mid and right */
    for (; i_out_y < dim_im_out - bottom_padding; i_out_y++)
    {

        /* left part */
        for (i_out_x = 0; i_out_x < left_padding; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - top_padding; i_ker_y < i_out_y * stride - top_padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - left_padding; i_ker_x < i_out_x * stride - left_padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        memset(pBuffer, 0, sizeof(int16_t) * ch_im_in);
                    }
                    else
                    {
                        ${config.reordered_no_shift_load_fn}(
% if config.in_data_t == 'u8':
                                                Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in,
% elif config.in_data_t == 'u4':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1),
% elif config.in_data_t == 'u2':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 2),
% endif
                                                pBuffer,
                                                ch_im_in,
                                                z_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut = ${config.nn_mat_mul_fn}(wt,
                                                bufferA,
                                                ch_im_out,
                                                ch_im_in * dim_kernel * dim_kernel,
                                                bias,
                                                pOut,
                                                z_wt,
% if config.folding == "thr":
                                                thresholds);
% else:
                                                z_out,
                                                m_zero,
                                                n_zero);
% endif
                /* counter reset */
                pBuffer = bufferA;
            }
        }

        /* mid part */
        for (; i_out_x < dim_im_out - right_padding; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - top_padding; i_ker_y < i_out_y * stride - top_padding + dim_kernel; i_ker_y++)
            {
                ${config.reordered_no_shift_load_fn}(
% if config.in_data_t == 'u8':
                                                Im_in + (i_ker_y * dim_im_in + i_out_x * stride - top_padding) * ch_im_in,
% elif config.in_data_t == 'u4':
                                                Im_in + (((i_ker_y * dim_im_in + i_out_x * stride - top_padding) * ch_im_in) >> 1),
% elif config.in_data_t == 'u2':
                                                Im_in + (((i_ker_y * dim_im_in + i_out_x * stride - top_padding) * ch_im_in) >> 2),
% endif
                                                pBuffer,
                                                ch_im_in * dim_kernel,
                                                z_in);
                pBuffer += ch_im_in * dim_kernel;
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut = ${config.nn_mat_mul_fn}(wt,
                                                bufferA,
                                                ch_im_out,
                                                ch_im_in * dim_kernel * dim_kernel,
                                                bias,
                                                pOut,
                                                z_wt,
% if config.folding == "thr":
                                                thresholds);
% else:
                                                z_out,
                                                m_zero,
                                                n_zero);
% endif
                /* counter reset */
                pBuffer = bufferA;
            }
        }

        /* right part */
        for (; i_out_x < dim_im_out; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - top_padding; i_ker_y < i_out_y * stride - top_padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - left_padding; i_ker_x < i_out_x * stride - left_padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        memset(pBuffer, 0, sizeof(int16_t) * ch_im_in);
                    }
                    else
                    {
                        ${config.reordered_no_shift_load_fn}(
% if config.in_data_t == 'u8':
                                                Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in,
% elif config.in_data_t == 'u4':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1),
% elif config.in_data_t == 'u2':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 2),
% endif
                                                pBuffer,
                                                ch_im_in,
                                                z_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut = ${config.nn_mat_mul_fn}(wt,
                                                bufferA,
                                                ch_im_out,
                                                ch_im_in * dim_kernel * dim_kernel,
                                                bias,
                                                pOut,
                                                z_wt,
% if config.folding == "thr":
                                                thresholds);
% else:
                                                z_out,
                                                m_zero,
                                                n_zero);
% endif
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    for (; i_out_y < dim_im_out; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out; i_out_x++)
        {
            /* This part implements the im2col function */
            for (i_ker_y = i_out_y * stride - top_padding; i_ker_y < i_out_y * stride - top_padding + dim_kernel; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride - left_padding; i_ker_x < i_out_x * stride - left_padding + dim_kernel; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in || i_ker_x < 0 || i_ker_x >= dim_im_in)
                    {
                        memset(pBuffer, 0, sizeof(int16_t) * ch_im_in);
                    }
                    else
                    {
                        ${config.reordered_no_shift_load_fn}(
% if config.in_data_t == 'u8':
                                                Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in,
% elif config.in_data_t == 'u4':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 1),
% elif config.in_data_t == 'u2':
                                                Im_in + (((i_ker_y * dim_im_in + i_ker_x) * ch_im_in) >> 2),
% endif
                                                pBuffer,
                                                ch_im_in,
                                                z_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel * dim_kernel)
            {
                pOut = ${config.nn_mat_mul_fn}(wt,
                                                bufferA,
                                                ch_im_out,
                                                ch_im_in * dim_kernel * dim_kernel,
                                                bias,
                                                pOut,
                                                z_wt,
% if config.folding == "thr":
                                                thresholds);
% else:
                                                z_out,
                                                m_zero,
                                                n_zero);
% endif
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    /* check if there is left-over for compute */
    $ { config.get_leftover_code() }

#else
#error "Cortex-M0 and Cortex-M3 not supported"
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
#endif /* ARM_MATH_DSP */

    /* Return to application */
    return ARM_MATH_SUCCESS;
}

/**
 * @} end of NNConv group
 */
