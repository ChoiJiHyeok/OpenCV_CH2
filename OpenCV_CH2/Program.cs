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

            Mat image = Cv2.ImRead(@"C:\Users\USER\Downloads\tomato.webp");
            Cv2.ImShow("gg",image);
        }

        private static void Event(int pos, IntPtr userdata) // callback 함수에 전달할 매개변수 함수
        {
            Mat color = new Mat(userdata);
            color.SetTo(new Scalar(pos, pos, pos));
            Cv2.ImShow("Pallete", color);
        }
    }
}
