// This is mul/vil2/tests/test_image_view_maths.cxx
#include <vcl_iostream.h>
#include <vcl_vector.h>
#include <vil2/vil2_image_view_maths.h>
#include <vil2/vil2_copy.h>
#include <testlib/testlib_test.h>

void test_image_view_maths_byte()
{
  vcl_cout << "*****************************\n";
  vcl_cout << " Testing vil2_image_view_maths\n";
  vcl_cout << "*****************************\n";

  int n=10, m=8;

  vil2_image_view<vil2_byte> imA(n,m,1);
  vil2_image_view<vil2_byte> imB(n,m,1);

  double sum2 = 0;
  for (int y=0;y<imA.nj();++y)
    for (int x=0;x<imA.ni();++x)
	{
      imA(x,y) = 1+x+y*10; sum2+= imA(x,y)*imA(x,y);
      imB(x,y) = 1+y+x*10;
    }

  double sum;
  vil2_sum(sum,imA,0);
  TEST_NEAR("Sum",sum,0.5*80*81,1e-8);

  double mean;
  vil2_mean(mean,imA,0);
  TEST_NEAR("mean",mean,0.5*80*81/80.0,1e-8);

  double sum_sq;
  vil2_sum_squares(sum,sum_sq,imA,0);
  TEST_NEAR("Sum of squares",sum_sq,sum2,1e-8);

  vil2_image_view<vil2_byte> imC = vil2_copy_deep(imA);
  vil2_scale_values(imC,2.0);
  TEST_NEAR("Values scaled",imC(3,5),imA(3,5)*2,1e-8);

  imC = vil2_copy_deep(imA);
  vil2_scale_and_offset_values(imC,2.0,7);
  TEST_NEAR("Values scaled+offset",imC(3,5),imA(3,5)*2+7,1e-8);


  vil2_image_view<float> im_sum;
  vil2_image_sum(imA,imB,im_sum);
  TEST("Width of im_sum",im_sum.ni(),imA.ni());
  TEST("Height of im_sum",im_sum.nj(),imA.nj());
  TEST_NEAR("im_sum(5,7)",im_sum(5,7),float(imA(5,7))+float(imB(5,7)),1e-6);

  vil2_image_view<float> im_diff;
  vil2_image_difference(imA,imB,im_diff);
  TEST_NEAR("im_diff(5,2)",im_diff(5,2),float(imA(5,2))-float(imB(5,2)),1e-6);

  vil2_image_view<float> im_abs_diff;
  vil2_image_abs_difference(imA,im_sum,im_abs_diff);
  TEST_NEAR("im_abs_diff(3,7)",im_abs_diff(3,7),vcl_fabs(float(imA(3,7))-float(im_sum(3,7))),1e-6);

  float is45 = im_sum(4,5);
  vil2_add_image_fraction(im_sum,0.77,imA,0.23);
  TEST_NEAR("add_fraction",im_sum(4,5),0.77*is45+0.23*imA(4,5),1e-5);
}

MAIN( test_image_view_maths )
{
  START( "vil2_image_view_maths" );

  test_image_view_maths_byte();

  SUMMARY();
}

