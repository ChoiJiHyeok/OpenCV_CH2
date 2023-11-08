using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace OpenCV_CH2
{
    class Program
    {
        static void Main(string[] args)
        {
            //    Mat src = new Mat(new OpenCvSharp.Size(500, 500), MatType.CV_8UC3, new Scalar(255, 255, 255));

            //    Cv2.ImShow("draw", src);
            //    MouseCallback cvMouseCallback = new MouseCallback(Event); // 마우스 이벤트가 발생 했을떄 전달 할 메서드 
            //    Cv2.SetMouseCallback("draw", cvMouseCallback, src.CvPtr); // 마우스 콜백 함수 설정
            //    Cv2.WaitKey(0);
            //    Cv2.DestroyAllWindows();
            //}
            //static void Event(MouseEventTypes @event, int x, int y, MouseEventFlags flags, IntPtr userdata) // 마우스 이벤트 핸들러 
            //{
            //    Mat data = new Mat(userdata);
            //    if(flags == MouseEventFlags.LButton)
            //    {
            //        Cv2.Circle(data, new OpenCvSharp.Point(x, y), 10, new Scalar(0, 0, 255), -1);
            //        Cv2.ImShow("draw", data);

            //    }


            //VideoCapture capture = new VideoCapture(@"C:\Users\USER\Downloads\production_id_4231736 (2160p).mp4"); // 객체 초기화
            //Mat frame = new Mat();

            //while (true)
            //{
            //    if(capture.PosFrames == capture.FrameCount) // PosFrame : 현재 프레임, FrameCount : 동영상의 총 프레임
            //    {
            //        capture.Open(@"C:\Users\USER\Downloads\production_id_4231736 (2160p).mp4"); // 동영상 파일 Open



            //    }
            //    capture.Read(frame);
            //    Cv2.ImShow("VideoFrame", frame);

            //    if (Cv2.WaitKey(33) == 'q')
            //    {
            //        break;
            //    }

            //}
            //frame.Dispose();
            //capture.Release(); // clolsed video file of device
            //Cv2.DestroyAllWindows();

            //VideoCapture videoCapture = new VideoCapture(0); // VideoCapture capture = new VideoCapture(int index)
            //Mat frame1 = new Mat();
            //videoCapture.Set(VideoCaptureProperties.FrameWidth, 640);  // Frame 너비 640
            //videoCapture.Set(VideoCaptureProperties.FrameHeight, 480); // Frame 높이 480

            //while(true)
            //{
            //    if(videoCapture.IsOpened() == true) //성공하면 true 반환, 실패하면 false 반환
            //    {
            //        videoCapture.Read(frame1);
            //        Cv2.ImShow("VideoFrame1", frame1);
            //        if(Cv2.WaitKey(33) == 'q')
            //        {
            //            break;
            //        }
            //    }
            //}
            //frame1.Dispose(); // 리소스 해제 
            //videoCapture.Release(); 
            //Cv2.DestroyAllWindows();

            //int value = 0;
            //Cv2.NamedWindow("Pallet");
            //Cv2.CreateTrackbar("Color", "Pallet", ref value, 255); // track bar 생성

            //while (true)
            //{
            //    int pixel = Cv2.GetTrackbarPos("Color", "Pallet");// track bar 위치 반환
            //    Mat src = new Mat(new OpenCvSharp.Size(500, 500), MatType.CV_8UC3, new Scalar(pixel, value, value)); 
            //    Cv2.ImShow("pallet", src);
            //    if (Cv2.WaitKey(33) == 'q')
            //    {
            //        break;
            //    }
            //}
            //Cv2.DestroyAllWindows();

            // track bar call back delegate 
            //TrackbarCallbackNative(int pos, IntPtr userData)

            //int value2 = 0;

            //Mat src2 = new Mat(new OpenCvSharp.Size(500, 500), MatType.CV_8UC3); // 크기 500x500 , 정밀도 8비트,usigned byte, 3채널 mat 객체 생성 
            //TrackbarCallbackNative trackbarCallback = new TrackbarCallbackNative(Event); // 트랙바 콜백함수에 Event 메서드 전달

            //Cv2.NamedWindow("Pallete");
            //Cv2.CreateTrackbar("Color", "Pallete", ref value2, 255, trackbarCallback, src2.CvPtr);
            //Cv2.WaitKey();
            //Cv2.DestroyAllWindows();  

            //Mat img = new Mat(new OpenCvSharp.Size(640, 480), MatType.CV_8UC3);
            //bool save;

            //ImageEncodingParam[] prms = new ImageEncodingParam[] { new ImageEncodingParam(ImwriteFlags.JpegQuality, 100), new ImageEncodingParam(ImwriteFlags.JpegProgressive, 1) };
            //save = Cv2.ImWrite(@"CV.jpeg", img, prms);
            //Console.WriteLine(save);

            //Cv2.VideoWriter(string fileName, FourCC fourcc, double fps, OpenCvSharp.Size frameSize, bool isColor = true); // 동영상 저장 함수

            //VideoCapture capture = new VideoCapture(@"C:\Users\USER\Desktop\Star.mp4"); // 경로에 있는 동영상 객체
            //Mat frame = new Mat(new OpenCvSharp.Size(capture.FrameWidth/2, capture.FrameHeight/2), MatType.CV_8UC3);// mat 객체
            //VideoWriter videoWriter = new VideoWriter(); // VideoWriter 객체
            //bool isWrite = false; //flag

            //while (true)
            //{
            //    if (capture.PosFrames == capture.FrameCount)
            //    {
            //        capture.Open(@"C:\Users\USER\Desktop\Star.mp4");
            //    }

            //    capture.Read(frame);

            //    Cv2.ImShow("VideoFrame", frame); 
            //    int key = Cv2.WaitKey(33); //33밀리세컨 단위의 시간을 키입력 대기
            //    if (key == 4) 
            //    {
            //        videoWriter.Open("Video.mp4", FourCC.XVID, 30, new OpenCvSharp.Size(frame.Width, frame.Height), true); //todo : avi형식으로는 저장이 예외가 발생하고, mp4로는 정상저장이 됨 
            //        isWrite = true;
            //    }
            //    else if (key == 24)
            //    {
            //        videoWriter.Release();
            //        isWrite = false;
            //    }
            //    else if (key == 'q')
            //    {
            //        break;
            //    }

            //    if (isWrite == true)
            //    {
            //        videoWriter.Write(frame);
            //    }    
            //}

            //videoWriter.Release();
            //capture.Release();
            //Cv2.DestroyAllWindows();

            //Cv2.CvtColor(Mat src, Mat dst, ColorConversionCodes code, int dstCn = 0); // 색상 공간 변환 함수

            //Mat src = Cv2.ImRead(@"C:\Users\USER\Downloads\bird.jpg"); // 입력 이미지
            //Mat dst = new Mat(src.Size(), MatType.CV_8UC1); // 출력 이미지

            //Cv2.CvtColor(src, dst, ColorConversionCodes.BGR2GRAY); // 색상 공간 변환 함수

            //Cv2.ImShow("dst", dst); // show img 12
            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();

            // 원본 이미지 색상 공간2결과 이미지 색상 공간

            //채널 분리 함수

            //Mat[] mv = Cv2.Split(Mat src);  // 채널 분리 함수
            //Cv2.Merge(Mat[] mv, Mat dst); // 채널 병합 함수

            //Cv2.InRange(Mat src, Scalar lowerb, Scalar upperb, Mat dst); // 배열 요소의 범위 설정 함수

            //Mat image = Cv2.ImRead(@"C:\Users\USER\Downloads\tomato.jpg"); // read image
            //Mat hsv = new Mat(image.Size(), MatType.CV_8UC3); // hsv 객체
            //Mat dst = new Mat(image.Size(), MatType.CV_8UC3); // 출력 객체

            //Cv2.CvtColor(image, hsv, ColorConversionCodes.BGR2HSV); // image 색상 공간 변환
            //Mat[] HSV = Cv2.Split(hsv); // 채널 분리
            //Mat H_orange = new Mat(image.Size(), MatType.CV_8UC1); //mat 객체
            //Cv2.InRange(HSV[0], new Scalar(8), new Scalar(20), H_orange); // 배열 요소의 범위 설정

            //Cv2.BitwiseAnd(hsv, hsv, dst, H_orange); // 마스크를 씌운다 ??
            //Cv2.CvtColor(dst, dst, ColorConversionCodes.HSV2BGR); // HSV 색상 공간을 다시 BGR로

            //Cv2.ImShow("orange", dst);
            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();

            //Cv2.AddWeighted(Mat src1, double alpha, Mat src2, double beta, double gamma, Mat dst, int dtype = 1);// 배열 병합 함수 

            //Mat pic = Cv2.ImRead(@"C:\Users\USER\Downloads\tomato.jpg"); // read image
            //Mat hsv = new Mat(pic.Size(), MatType.CV_8UC3);
            //Mat lower_red = new Mat(pic.Size(), MatType.CV_8UC3);
            //Mat upper_red = new Mat(pic.Size(), MatType.CV_8UC3);
            //Mat added_red = new Mat(pic.Size(), MatType.CV_8UC3);
            //Mat dst = new Mat(pic.Size(), MatType.CV_8UC3);

            //Cv2.CvtColor(pic, hsv, ColorConversionCodes.BGR2HSV);// pic를 hsv로 반환, bgr to hsv 

            //Cv2.InRange(hsv, new Scalar(0, 100, 100), new Scalar(5, 255, 255), lower_red); // hsv를 scalar(0, 100, 100), scalar(5, 255, 255)사이의 요소를 검출해 lower_red로 반환
            //Cv2.InRange(hsv, new Scalar(170, 100, 100), new Scalar(179, 255, 255), upper_red); // hsv를 scalar(0, 100, 100), scalar(5, 255, 255)사이의 요소를 검출해 upper_red로 반환
            //Cv2.AddWeighted(lower_red, 1.0, upper_red, 1.0, 0.0, added_red); // 배열 병합 : lower_red*1.0 + upper_red*1.0 + gamma(0.0)

            //Cv2.BitwiseAnd(hsv, hsv, dst, added_red);
            //Cv2.CvtColor(dst, dst, ColorConversionCodes.HSV2BGR); // dst를 dst로 반환 , HSV to BGR

            //Cv2.ImShow("dst", dst);
            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();

            //Cv2.Threshold(Mat src, Mat dst, double thresh, double maxval, ThresholdTypes type); // 이진화 함수 


            Mat picture = Cv2.ImRead(@"C:\Users\USER\Downloads\swan.jpg"); // read image
            Mat gray = new Mat(picture.Size(), MatType.CV_8UC1);
            Mat binary = new Mat(picture.Size(), MatType.CV_8UC1);

            Cv2.CvtColor(picture, gray, ColorConversionCodes.BGR2GRAY); // bgr to gray, picture를 gray로 반환
            //Cv2.Threshold(gray, binary, 127, 255, ThresholdTypes.Otsu); // 이진화, 임계값 : 127, 최대값 : 255 gray를 이진화 해서 변형해서 binary에 저장, Otsu type
            Cv2.Threshold(gray, binary, 0, 255, ThresholdTypes.Triangle); // 최적의 임계값 계산, 삼각형 알고리즘 적용

            double thresholdValue = Cv2.Threshold(gray, new Mat(), 0, 255, ThresholdTypes.Triangle); // theshold 계산
            Console.WriteLine(thresholdValue.ToString()); 

            Cv2.ImShow("binary", binary);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();

            //적응형 이진화(adaptive binarization)

            //Cv2.AdaptiveThreshold(Mat src, Mat dst, double maxValue, AdaptiveThresholdTypes adaptiveMethod, ThresholdTypes thresholdType, int bolckSize, double c);

            //Mat picture = Cv2.ImRead(@"C:\Users\USER\Downloads\swan.jpg"); // read image
            //Mat gray = new Mat(picture.Size(), MatType.CV_8UC1);
            //Mat binary = new Mat(picture.Size(), MatType.CV_8UC1);

            //Cv2.CvtColor(picture, gray, ColorConversionCodes.BGR2GRAY); // bgr to gray, picture를 gray로 반환
            //Cv2.AdaptiveThreshold(gray, binary, 255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.Binary, 25, 5); // 

            //Cv2.ImShow("binary", binary);
            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();

            // 이미지 연산
            //Cv2.Add(Mat src1, Mat src2, Mat dst, Mat mask = null, int dtype = -1);// 덧셈 함수

            //Cv2.Subtract(Mat src1, Mat src2, Mat dst, Mat mask = null, int dtype = -1);// 뺄샘 함수

            //Cv2.Multiply(Mat src1, Mat src2, Mat dst, double scale = 1, int dtype = -1);// 곱셈 함수

            //Cv2.Divide(Mat src1, Mat src2, Mat dst, double scale = 1, int dtype = -1);// 나눗셈 함수

            //Cv2.Max(Mat src1, Mat src2, Mat dst);// 최댓값 함수

            //Cv2.Min(Mat src1, Mat src2, Mat dst);// 최솟값 함수

            //Cv2.MinMaxLoc(Mat src, out double minVal, out double maxVal, out Point minLoc, out Point maxLoc); //최소 최대 위치 반환 함수

            //Cv2.Abs(Mat src); // 절대값 함수

            //Cv2.Absdiff(Mat src1, Mat src2, Mat dst); // 절대값 차이 함수

            //Cv2.Compare(Mat src1, Mat src2, Mat dst, CmpType cmpop); // 비교 함수

            //success = Cv2.Solve(Mat src1, Mat src2, Mat dst, DecompTypes.LU); // 선형 방정식 시스템의 해 찾기 함수

            //Cv2.BitwiseAnd(Mat src1, Mat src2, Mat dst, Mat mask = null); // AND 연산 함수

            //Cv2.BitwiseOr(Mat src1, Mat src2, Mat dst, Mat mask = null); // OR 연산 함수

            //Cv2.BitwiseXor(Mat src1, Mat src2, Mat dst, Mat mask = null); // XOR 연산 함수

            //Cv2.BitwiseNot(Mat src1, Mat src2, Mat dst, Mat mask = null); // NOT 연산 함수

            ///
            ///Example
            ///
            //Mat src1 = Cv2.ImRead(@"C:\Users\USER\Downloads\tomato.jpg");
            //Mat dst = new Mat(src1.Size(), MatType.CV_8UC3);

            //Cv2.Compare(src1, new Scalar(200, 127, 100), dst, CmpType.GT); /* 비교연산, 배열과 스칼라의 요소별 비교 연산, src1이 src2보다 요소가 큼, src1의 BGR요소의 값이 200, 127, 100보다 큰 경우
            //요소값 유지 나머지 요소는 0으로 변경 */

            //Cv2.ImShow("dst", dst);
            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();


            //흐림효과
            //Cv2.Blur(Mat src, Mat dst, OpenCvSharp.Size ksize, OpenCvSharp.Point? anchor = null, BorderTypes borderType = BorderTypes.Reflect101); // 단순 흐림 효과

            //Cv2.BoxFilter(Mat src, Mat dst, MatType ddepth, OpenCvSharp.Size ksize, OpenCvSharp.Point? anchor = null, bool normalize = true, BorderTypes borderType = BorderTypes.Reflect101); // 박스 필터 흐림 효과

            ///
            ///normalize = true 일경우 정규화된 박스 필터(normalized filter)로 변경, 모든 값이 커널의 개수(면적)만큼 나눠진다.
            ///

            //Cv2.MedianBlur(Mat src, Mat dst, OpenCvSharp.Size ksize);  // 중간값 흐림 효과
            //Cv2.GaussianBlur(Mat src, Mat dst, OpenCvSharp.Size ksize, double sigmaX, double sigmaY, BorderTypes borderType = BorderTypes.Reflect101); // 가우시안 흐림 효과

            //Cv2.BilateralFilter(Mat src, Mat dst, int d, double sigmaColor, double sigmaSpace, BorderTypes borderType = BorderTypes.Reflect101); // 양방향 필터 효과


            //Mat src = Cv2.ImRead(@"C:\Users\USER\Downloads\tomato.jpg");
            //Mat dst = new Mat(src.Size(), MatType.CV_8UC3);

            //Cv2.GaussianBlur(src, dst, new OpenCvSharp.Size(9, 9), 3, 3, BorderTypes.Isolated);//kernel 크기 9x9, x방향 가우스 커널 표준 편차: 3, y방향 가우스 커널 표준 편차: 3 

            //Cv2.ImShow("dst", dst);
            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();

            //Cv2.PyrUp(Mat src, Mat dst, OpenCvSharp.Size ? dstSize = null, BorderTypes borderType = BorderTypes.Reflect101); // 이미지 확대

            //Cv2.PyrDown(Mat src, Mat dst, OpenCvSharp.Size ? dstSize = null, BorderTypes borderType = BorderTypes.Reflect101); // 이미지 축소

            //Mat src = Cv2.ImRead(@"C:\Users\USER\Downloads\tomato.jpg");

            //Mat dst = new Mat(src.Size(), MatType.CV_8UC3);

            ////Cv2.PyrUp(src, dst, new OpenCvSharp.Size(src.Width*2+1, src.Height*2-1)); //  src.Width*2+1, src.Height*2-1만큼 이미지 확대
            //Cv2.PyrDown(src, dst, new OpenCvSharp.Size(src.Width/2,src.Height/2));
            //Cv2.ImShow("dst", dst);
            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();

            // 이미지 크기 조절
            //Cv2.Resize(Mat src, Mat dst, OpenCvSharp.Size dsize, double fx = 0, double fy = 0, InterpolationFlags interpolation = InterpolationFlags.Linear);

            //Mat picture = Cv2.ImRead(@"C:\Users\USER\Downloads\tomato.jpg");
            //Mat dst = new Mat(new OpenCvSharp.Size(1, 1), MatType.CV_8UC3);

            //dst = picture.SubMat(280, 310, 240, 405);
            //Cv2.Resize(dst, dst, new OpenCvSharp.Size(9999, 0), 2.0, 2.0, InterpolationFlags.Cubic);

            //Cv2.ImShow("dst", dst);
            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();



            // 대칭 회전
            //Cv2.Flip(Mat src, Mat dst, FlipMode flipcode);  // 대칭 함수

            // 회전 행렬

            //Mat matrix = Cv2.GetRotationMatrix2D(Point2f center, double angle, double scale); // Center 중심좌표, scale 비율, angle 회전 각도 

            //Mat src = Cv2.ImRead(@"C:\Users\USER\Downloads\glass.jpg");
            //Mat dst = new Mat();
            //Mat dst1= new Mat(new OpenCvSharp.Size(src.Width, src.Height), MatType.CV_8UC3);


            //Cv2.Flip(src, dst, FlipMode.Y); // y축 기준 대칭

            //Mat matrix = Cv2.GetRotationMatrix2D(new Point2f(src.Width / 2, src.Height / 2), 90.0, 1.0); // 이미지 중심점 기준, 1비율, 반시계 방향 90도 회전

            //Cv2.WarpAffine(dst, dst, matrix, new Size(src.Width, src.Height));

            //Cv2.ImShow("dst", dst);
            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();

            //Mat M = Cv2.GetAffineTransform(IEnumerable<Point2f> src, IEnumerable<Point2f> dst);// 아핀 맵 행렬 생성 함수

            //Cv2.WarpAffine(Mat src, Mat dst, Mat m, OpenCvSharp.Size dsize, InterpolationFlags flags = InterpolationFlags.Linear, BorderTypes borderMode = BorderTypes.Constant, Scalar ? borderValue = null); // 아핀 변환 함수

            // 원근 변환
            //Mat m = Cv2.GetPerspectiveTransform(IEnumerable < Point2f > src, IEnumerable < Point2f > dst); // 원근맵 행렬 생성 함수

            //Cv2.WarpPerspective(Mat src, Mat dst, Mat M, OpenCvSharp.Size dsize, InterpolationFlags flags = InterpolationFlags.Linear, BorderTypes borderMode = BorderTypes.Constant, Scalar ? borderValue = null); // 원근 변환 함수

            //Mat src = Cv2.ImRead(@"C:\Users\USER\Downloads\cloud.jpg");
            //Mat dst = new Mat(new OpenCvSharp.Size(src.Width, src.Height), MatType.CV_8UC3);

            //List<Point2f> src_pts = new List<Point2f>() // 변환전 세개의 픽셀 좌표
            //{
            //    new Point2f(0.0f, 0.0f),
            //    new Point2f(0.0f, src.Height),
            //    new Point2f(src.Width, src.Height)
            //};

            // List<Point2f> dst_pts = new List<Point2f>()  // 변환후 세개의 픽셀 좌표
            //{
            //    new Point2f(300.0f, 300.0f),
            //    new Point2f(300.0f, src.Height),
            //    new Point2f(src.Width - 400.0f, src.Height-200.0f)
            //};

            //Mat M = Cv2.GetAffineTransform(src_pts, dst_pts); // 아핀 맵 행렬 생성

            //Cv2.WarpAffine(src, dst, M, new OpenCvSharp.Size(src.Width, src.Height), borderValue: new Scalar(127, 127, 127, 0)); 
            ////아핀 변환 함수, 입력 이미지(src)에 아핀맵 행렬 적용(M), 출력 이미지 크기(dsize)고 변형해 출력 이미지(dst)로 반환, 테두리 색상(borderValue)은 변환 후 발생한 공간에 할당할 색상

            //Cv2.ImShow("dst", dst);
            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();

            // morphological
            /* 모폴로지 변환(morphologiacal tranformations)은 영상이나 이미지를 형태학적 적급하는 기법
             * 주로 영상내 픽셀값 대체에 사용
             * 오이즈 제거, 요소 결합 및 분리, 강도 피크 검출등에 이용
             * 집합의 포함 관계, 이동(translation), 대칭(reflection), 여집합(complement), 차집합(difference)등의 성질 이용
             * 기본 변환으로 팽창(dilation)과 침식(erosion)이 있음
             * 팽창 : 커널(kernel) 영역 안에 존재하는 모든 픽셀의 값을 커널 내부의 극댓값(local maxium)으로 대체, 구조 요소(element)를 활용해 이웃한 픽셀을 최대 픽셀값으로 대체, 
             * 어두운 영역이 줄고 밝은 영역이 늘어남,
             * 커널 크기가 반복 횟수에 따라 스펙클(speckle)이 커지며 객체 내부의 홀(hole)이 사라진다.
             * 침식 : 커널 영역 안에 존재하는 모든 픽셀의 값을 커널 내부의 극솟값(local minimum)으로 대체, 구조 요소를 활용해 이웃한 픽셀은 최소 픽셀값으로대체,
             * 밝은 영역이 줄어들고 어두운 영역이 늘어남,
             * speckle이 사라지며, hole이 커짐
             */

            ///
            /// 구조 요소 생성 함수
            ///
            //Mat kernel = Cv2.GetStructuringElement(MorphShapes shape, new OpenCvSharp.Size ksize, new OpenCvSharp.Point anchor); 

            /* 커널의 형태(shape)를 설정, 
             * 직사각형(Rect), 십자가(Cross), 타원(Ellipse) 모양으로 구조 요소 생성
             * anchor(고정점)은 필수 매개 변수 아님*/

            // 팽창 함수
            //Cv2.Dilate(Mat src, Mat dst, Mat kernel, new OpenCvSharp.Point? anchor = null, int iterations = 1, BorderTypes borderType = BorderTypes.Constant, Scalar? borderValue = null);

            //침식 함수
            //Cv2.Erode(Mat src, Mat dst, Mat kernel, Point ? anchor = null, int iterations = 1, BorderTypes borderType = BorderTypes.Constant, Scalar ? borderValue = null);       

            //Mat src = Cv2.ImRead(@"C:\Users\USER\Downloads\flower.jpg");
            //Mat dst = new Mat();

            //Mat kernel = Cv2.GetStructuringElement(MorphShapes.Cross, new OpenCvSharp.Size(7, 7)); // kernel : 십자가 구조요소, 7x7
            //Cv2.Dilate(src, dst, kernel, new OpenCvSharp.Point(-1, -1), 3, BorderTypes.Reflect101, new Scalar(0)); //팽창 , anchor : (-1, -1), iteration(팽창) : 3회, 테투리 색상(borderValue) : 이중픽셀X, 반사해서 확장

            //Cv2.ImShow("dst", dst);
            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();

            // 모폴로지 연산 

            // 팽창과 침식을 기본 연산으로 사용

            //모폴로지 연산 함수
            //Mat dst1 = Cv2.MorphologyEx(Mat src, Mat dst, MorphTypes op, Mat kernel, Point ? anchor, int iterations = 1, BorderTypes borderType = BorderTypes.Constant, Scalar ? borderValue = null);


            //Mat picture = Cv2.ImRead(@"C:\Users\USER\Downloads\dandelion.jpg", ImreadModes.Grayscale); // 이미지를 그레이 스케일로 읽기
            //Mat dst = new Mat();

            //Mat kernel = Mat.Zeros(new OpenCvSharp.Size(7,7), MatType.CV_8UC1); // kernel 초기화, 7x7
            //kernel[0, 7, 0, 1] = Mat.Ones(new OpenCvSharp.Size(1, 7), MatType.CV_8UC1); 
            //kernel[0, 1, 0, 7] = Mat.Ones(new OpenCvSharp.Size(7, 1), MatType.CV_8UC1);
            // // 첫번째 행과 열에 1의 요소를 할당
            //Cv2.MorphologyEx(picture, dst, MorphTypes.HitMiss, kernel, iterations: 10);

            //Cv2.ImShow("dst", dst);
            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();

            // 소벨 연산
            //Cv2.Sobel(Mat src, Mat dst, MatType ddepth, int xorder, int yorder, int ksize = 3, double scale = 1, double delta = 0, BorderTypes borderType = BorderTypes.Reflect101)

            //Cv2.Scharr(Mat src, Mat dst, MatType ddepth, int xorder, int yorder, double scale, double delta, BorderTypes borderType = BorderTypes.Reflect101);//  샤르 연산 함수

            // 라플라시안

            //Cv2.Laplacian(Mat src, Mat dst, MatType ddepth, int ksize, double scale, double delta, BorderTypes borderType = BorderTypes.Reflect101) // 라플라시안 함수

            // Canny Edge
            //Cv2.Canny(Mat src, Mat dst, double threshold1, double threshold2, int apertureSize = 3, bool L2gradient = false) // 캐니 엣지 함수

            //Mat src = Cv2.ImRead(@"C:\Users\USER\Downloads\book.jpg", ImreadModes.Grayscale);
            //Mat dst = new Mat();

            //Cv2.Sobel(src, dst, MatType.CV_8UC1, 1, 0, 3, 1, 0, BorderTypes.Reflect101); // 소벨 미분, x축 방향 1차 미분, y방향 미분x, kernel size = 3, scale(비율) = 1, delta(offset) = 0, 테두리 외삽법은 기본

            //Cv2.ImShow("dst", dst);

            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();

        }

        //private static void Event(int pos, IntPtr userdata) // callback 함수에 전달할 매개변수 함수
        //{
        //    Mat color = new Mat(userdata);
        //    color.SetTo(new Scalar(pos, pos, pos));
        //    Cv2.ImShow("Pallete", color);
        //}
    }
}
