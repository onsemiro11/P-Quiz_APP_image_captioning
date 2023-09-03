import 'dart:io';

import 'package:flutter/material.dart';
import 'package:install_test/ml_api/MlApi.dart';
import 'package:install_test/pages/photo_quiz.dart';

class Loading extends StatefulWidget {
  const Loading({Key? key, required this.ipath}) : super(key: key);
  final String ipath;

  @override
  _LoadingState createState() => _LoadingState();
}

class _LoadingState extends State<Loading> {
  var data;
  String text = 'Generating Quiz...';
  @override
  void initState() {
    super.initState();
    getLocation();
  }

  void getLocation() async {
    final File? file = File(widget.ipath); // 유효한 이미지인 경우 File 객체로 변환
    print(widget.ipath);
    //server로 이미지 업로드
    await uploadImage(file!, RequestUrl);
    print('pass upload');
    data = await getCaption(RequestUrl);

    text = data['captions'];

    Navigator.push(context, MaterialPageRoute(builder: (context) {
      return PhotoQuiz(ipath: widget.ipath, caption: text);
    }));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFF4E3E3),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              decoration: BoxDecoration(
                color: Color(0xFFF4E3E3),
                borderRadius: BorderRadius.circular(15.0), // 사진 모서리 둥글게
              ),
              child: Image.file(
                File(widget.ipath),
                width: 200,
                height: 200,
                fit: BoxFit.cover,
              ),
            ),
            SizedBox(height: 32),
            Text(
              'Generating Quiz!',
              style: TextStyle(
                  color: Color(0xFF2F3542),
                  fontSize: 32,
                  fontWeight: FontWeight.w700),
            ),
            SizedBox(height: 14),
            Text(
              'Generating a quiz\nwith the picture you put in.\nPlease wait a few minutes.',
              style: TextStyle(
                color: Color(0xFF2F3542),
                fontSize: 16,
              ),
            ),
            SizedBox(height: 27),
            Image.asset(
              'assets/images/loading.gif',
              width: 65,
              height: 65,
            ),
          ],
        ),
      ),
    );
  }
}
