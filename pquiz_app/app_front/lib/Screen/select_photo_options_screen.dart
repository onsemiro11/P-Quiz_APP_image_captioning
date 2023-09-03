import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../widgets/re_usable_select_photo_button.dart';

class SelectPhotoOptionsScreen extends StatelessWidget {
  // Function(ImageSource source)는 사진 선택 옵션을 터치할 때 호출되는 콜백 함수의 형식
  final Function(ImageSource source) onTap;

  // const는 불변성을 가지는 생성자
  // SelectPhotoOptionsScreen 위젯 생성시 초기화 작업 수행
  // 위젯 생성 시 onTap 콜백 함수를 전달하며, 사용자의 터치 동작에 응답하는데 사용
  const SelectPhotoOptionsScreen({
    Key? key,
    required this.onTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // 모달식 내용을 감싸는 컨테이너
    return Container(
      height: 210,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Color(0xFFF4E3E3),
        borderRadius: BorderRadius.circular(10),
      ),
      /*컨텐츠 주위의 간격 지정*/
      child: Stack(
        /*여러 위젯을 겹쳐서 표시할 수 있게 하는 위젯*/
        alignment: AlignmentDirectional.topCenter,
        clipBehavior: Clip.none,
        children: [
          Positioned(
            top: -35,
            child: Container(
              width: 50,
              height: 6,
              margin: const EdgeInsets.only(bottom: 20),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(2.5),
                color: Color(0xFFF4E3E3),
              ),
            ),
          ),
          const SizedBox(
            height: 20,
          ),
          Column(children: [
            SelectPhoto(
              onTap: () => onTap(ImageSource.gallery),
              icon: Icons.image,
              textLabel: 'Select a photo',
            ),
            const SizedBox(
              height: 10,
            ),
            const Center(
              child: Text(
                'OR',
                style: TextStyle(fontSize: 17),
              ),
            ),
            const SizedBox(
              height: 10,
            ),
            SelectPhoto(
              onTap: () => onTap(ImageSource.camera),
              icon: Icons.camera_alt_outlined,
              textLabel: 'Take a photo',
            ),
          ])
        ],
      ),
    );
  }
}
