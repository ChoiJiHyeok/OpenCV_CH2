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


            //Mat picture = Cv2.ImRead(@"C:\Users\USER\Downloads\swan.jpg"); // read image
            //Mat gray = new Mat(picture.Size(), MatType.CV_8UC1);
            //Mat binary = new Mat(picture.Size(), MatType.CV_8UC1);

            //Cv2.CvtColor(picture, gray, ColorConversionCodes.BGR2GRAY); // bgr to gray, picture를 gray로 반환
            //Cv2.Threshold(gray, binary, 127, 255, ThresholdTypes.Otsu); // 이진화, 임계값 : 127, 최대값 : 255 gray를 이진화 해서 변형해서 binary에 저장, Otsu type

            //Cv2.ImShow("binary", binary);
            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();

            //적응형 이진화(adaptive binarization)

            //Cv2.AdaptiveThreshold(Mat src, Mat dst, double maxValue, AdaptiveThresholdTypes adaptiveMethod, ThresholdTypes thresholdType, int bolckSize, double c);

            Mat picture = Cv2.ImRead(@"C:\Users\USER\Downloads\swan.jpg"); // read image
            Mat gray = new Mat(picture.Size(), MatType.CV_8UC1);
            Mat binary = new Mat(picture.Size(), MatType.CV_8UC1);

            Cv2.CvtColor(picture, gray, ColorConversionCodes.BGR2GRAY); // bgr to gray, picture를 gray로 반환
            Cv2.AdaptiveThreshold(gray, binary, 255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.Binary, 25, 5); // 

            Cv2.ImShow("binary", binary);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();

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

        }

        private static void Event(int pos, IntPtr userdata) // callback 함수에 전달할 매개변수 함수
        {
            Mat color = new Mat(userdata);
            color.SetTo(new Scalar(pos, pos, pos));
            Cv2.ImShow("Pallete", color);
        }
    }
}
