/* hello_opencl.c
 *
 * This is a simple OpenCL example for Parallella that performs a
 * matrix-vector multiplication on the Epiphany processor.
 *
 * THIS FILE ONLY is placed in the public domain by Brown Deer Technology, LLC.
 * in January 2013. No copyright is claimed, and you may use it for any purpose
 * you like. There is ABSOLUTELY NO WARRANTY for any purpose, expressed or
 * implied, and any warranty of any kind is explicitly disclaimed.  This
 * statement applies ONLY TO THE COMPUTER SOURCE CODE IN THIS FILE and does
 * not extend to any other software, source code, documentation, or any other
 * materials in which it may be included, or with which it is co-distributed.
 */

/* DAR */

/* AOLOFSSON: Comments and tweaks */

//#define DEVICE_TYPE	CL_DEVICE_TYPE_CPU
#define DEVICE_TYPE	CL_DEVICE_TYPE_ACCELERATOR

#define MAX_SOURCE_SIZE (0x100000)

#define PLAINTEXT_LENGTH 72

#define ALGORITHM_NAME "Bitmessage Proof-of-Work"

typedef struct {
  unsigned long target;
  char v[PLAINTEXT_LENGTH+1];
} sha512_key;

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CL/cl.h>

void handle_clerror(cl_int cl_error, const char *message, const char *file, int line);
#define HANDLE_CLERROR(cl_error, message) (handle_clerror(cl_error,message,__FILE__,__LINE__))

cl_int clerr;

int main()
{
   int i,j;
   int err;
   char buffer[256];

   unsigned int n = 16;

   cl_uint nplatforms;
   cl_platform_id* platforms;
   cl_platform_id platform;
   //---------------------------------------------------------
   //Discover and initialize the platform
   //---------------------------------------------------------
   clGetPlatformIDs( 0,0,&nplatforms);
   platforms = (cl_platform_id*)malloc(nplatforms*sizeof(cl_platform_id));
   clGetPlatformIDs( nplatforms, platforms, 0);

   for(i=0; i<nplatforms; i++) {
      platform = platforms[i];
      clGetPlatformInfo(platforms[i],CL_PLATFORM_NAME,256,buffer,0);
      if (!strcmp(buffer,"coprthr")) break;
   }

   if (i<nplatforms) platform = platforms[i];
   else exit(1);
   //---------------------------------------------------------
   //Discover and initialize the devices
   //---------------------------------------------------------
   cl_uint ndevices;
   cl_device_id* devices;
   cl_device_id dev;

   clGetDeviceIDs(platform,DEVICE_TYPE,0,0,&ndevices);
   devices = (cl_device_id*)malloc(ndevices*sizeof(cl_device_id));
   clGetDeviceIDs(platform, DEVICE_TYPE,ndevices,devices,0);

   if (ndevices) dev = devices[0];
   else exit(1);


   //---------------------------------------------------------
   //Create a context
   //---------------------------------------------------------
   cl_context_properties ctxprop[3] = {
      (cl_context_properties)CL_CONTEXT_PLATFORM,
      (cl_context_properties)platform,
      (cl_context_properties)0
   };
   cl_context ctx = clCreateContext(ctxprop,1,&dev,0,0,&err);

   //---------------------------------------------------------
   //Create a command queue
   //---------------------------------------------------------
   cl_command_queue cmdq = clCreateCommandQueue(ctx,dev,0,&err);

   //---------------------------------------------------------
   //Allocate dynamic memory on the host
   //---------------------------------------------------------
   size_t a_sz = n*n*sizeof(float);
   size_t b_sz = n*sizeof(float);
   size_t c_sz = n*sizeof(float);

   static size_t hash_sz = sizeof(sha512_key);
   static size_t dest_sz = sizeof(unsigned long);

   float* a = (float*)malloc(n*n*sizeof(float));
   float* b = (float*)malloc(n*sizeof(float));
   float* c = (float*)malloc(n*sizeof(float));
   for(i=0;i<n;i++) for(j=0;j<n;j++) a[i*n+j] = 1.1f*i*j;
   for(i=0;i<n;i++) b[i] = 2.2f*i;
   for(i=0;i<n;i++) c[i] = 0.0f;

   char *input = "00000000000000003758f55b5a8d902fd3597e4ce6a2d3f23daff735f65d9698c270987f4e67ad590b93f3ffeba0ef2fd08a8dc2f87b68ae5a0dc819ab57f22ad2c4c9c8618a43b3";

   unsigned long startpos = 0;

   //---------------------------------------------------------
   //Copy data to device buffer
   //---------------------------------------------------------
   cl_mem a_buf = clCreateBuffer(ctx,CL_MEM_USE_HOST_PTR,a_sz,a,&err);
   cl_mem b_buf = clCreateBuffer(ctx,CL_MEM_USE_HOST_PTR,b_sz,b,&err);
   cl_mem c_buf = clCreateBuffer(ctx,CL_MEM_USE_HOST_PTR,c_sz,c,&err);

   cl_mem hash_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, hash_sz, input, &clerr);
   /* HANDLE_CLERROR(clerr,"Error while allocating memory for hash buffer."); */
   cl_mem dest_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, dest_sz, NULL, &clerr);
   /* HANDLE_CLERROR(clerr,"Error while allocating memory for return buffer."); */

   //---------------------------------------------------------
   //The kernel
   //---------------------------------------------------------
   FILE *fp;
   const char fileName[] = "./kernel.cl";
   char* src;
   size_t src_sz;

   /* Load kernel source file */
   fp = fopen(fileName, "r");
   if (!fp) {
      fprintf(stderr, "Failed to load kernel.\n");
      exit(1);
   }
   src = (char *)malloc(MAX_SOURCE_SIZE);
   src_sz = fread(src, 1, MAX_SOURCE_SIZE, fp);
   fclose(fp);

   //---------------------------------------------------------
   //Compiling the kernel
   //---------------------------------------------------------
   cl_program prg = clCreateProgramWithSource(ctx,1,(const char**)&src,&src_sz,&err);

   clBuildProgram(prg,1,&dev,0,0,0);

   cl_kernel krn = clCreateKernel(prg,"kernel_sha512",&err);
   /* cl_kernel krn = clCreateKernel(prg,"matvecmult_kern",&err); */

   //---------------------------------------------------------
   //Set kernel arguments
   //---------------------------------------------------------
   clSetKernelArg(krn,0,sizeof(cl_mem),&hash_buf);
   clSetKernelArg(krn,1,sizeof(cl_mem),&dest_buf);
   clSetKernelArg(krn,2,sizeof(cl_uint),&startpos);

   //---------------------------------------------------------
   //Queue up kernel for execution
   //---------------------------------------------------------
   size_t gtdsz[] = { n };
   size_t ltdsz[] = { 16 };
   cl_event ev[10];
   clEnqueueNDRangeKernel(cmdq,krn,1,0,gtdsz,ltdsz,0,0,&ev[0]);


   //---------------------------------------------------------
   //Readb back result data
   //---------------------------------------------------------
   clEnqueueReadBuffer(cmdq,c_buf,CL_TRUE,0,c_sz,c,0,0,&ev[1]);
   err = clWaitForEvents(2,ev);

   //---------------------------------------------------------
   //Print result
   //---------------------------------------------------------
   for(i=0;i<n;i++) printf("c[%d] %f\n",i,c[i]);

   //---------------------------------------------------------
   //Release OpenCL resources
   //---------------------------------------------------------
   clReleaseEvent(ev[1]);
   clReleaseEvent(ev[0]);
   clReleaseKernel(krn);
   clReleaseProgram(prg);
   clReleaseMemObject(a_buf);
   clReleaseMemObject(b_buf);
   clReleaseMemObject(c_buf);
   clReleaseMemObject(hash_buf);
   clReleaseMemObject(dest_buf);
   clReleaseCommandQueue(cmdq);
   clReleaseContext(ctx);

   //---------------------------------------------------------
   //Free host resourcdes
   //---------------------------------------------------------
   free(a);
   free(b);
   free(c);

   return 0;
}
