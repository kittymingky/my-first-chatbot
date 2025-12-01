import os
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

# 환경 변수 로드
load_dotenv()

# Azure OpenAI 설정 (.env 파일의 1-3줄 형식 사용)
azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OAI_KEY")
azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT", "gpt-4o")
azure_oai_api_version = os.getenv("AZURE_OAI_API_VERSION", "2024-02-01")

# 넷플릭스 인기 캐릭터 데이터베이스
NETFLIX_CHARACTERS = {
    "오징어 게임": {
        "성기훈": {
            "성격": "도박 중독에 빠진 전직 자동차 공장 직공. 실패한 사업가이지만 가족에 대한 사랑이 깊고, 생존을 위해 끝까지 싸우는 강인한 의지를 가진 인물이에요!",
            "특징": "인간적이고 현실적인 모습을 보여주며, 극한 상황에서도 동료를 배려하는 따뜻한 마음을 가지고 있어요.",
            "명대사": "우리는 모두 같은 배를 탔어"
        },
        "조상우": {
            "성격": "서울대 출신의 똑똑한 투자 전문가. 냉정하고 계산적이지만, 마지막 순간에는 인간미를 보여주는 복합적인 캐릭터예요!",
            "특징": "높은 지능과 전략적 사고를 가지고 있지만, 인간관계에서는 냉담한 편이에요.",
            "명대사": "이 게임은 공정하지 않아"
        }
    },
    "킹덤": {
        "이창": {
            "성격": "조선의 왕세자로, 백성들을 위해 목숨을 걸고 싸우는 정의로운 인물이에요! 용감하고 지혜로우며, 아버지에 대한 사랑이 깊어요.",
            "특징": "무예에 뛰어나고, 백성들의 고통을 이해하며 함께 고민하는 진정한 리더예요.",
            "명대사": "백성들이 굶주리고 있는데, 나는 무엇을 하고 있었나"
        },
        "서비": {
            "성격": "의녀 출신으로, 좀비 바이러스의 치료법을 찾기 위해 고군분투하는 똑똑하고 용감한 여성 캐릭터예요!",
            "특징": "의술에 뛰어나고, 위험을 무릅쓰고도 진실을 추구하는 강인한 정신력을 가지고 있어요.",
            "명대사": "모든 생명은 소중해요"
        }
    },
    "사이버펑크: 엣지러너": {
        "데이비드 마르티네스": {
            "성격": "나이틀 시티에서 꿈을 쫓는 젊은이. 어머니의 죽음 후 사이버펑크가 되어 살아가지만, 여전히 순수한 마음을 간직하고 있어요!",
            "특징": "강한 의지와 동료에 대한 충성심이 있으며, 자신의 신념을 위해 끝까지 싸우는 인물이에요.",
            "명대사": "나는 특별해지고 싶어"
        }
    },
    "기묘한 이야기": {
        "일레븐": {
            "성격": "초능력을 가진 소녀로, 처음에는 말이 없고 조용하지만 친구들과의 우정을 통해 점점 밝아지는 캐릭터예요!",
            "특징": "강력한 초능력을 가지고 있지만, 평범한 삶을 꿈꾸는 순수한 소녀예요.",
            "명대사": "Friends don't lie"
        },
        "마이크 휠러": {
            "성격": "리더십이 있고, 친구들을 위해 항상 앞장서는 용감한 소년이에요!",
            "특징": "논리적이고 차분하며, 위기 상황에서도 침착하게 판단하는 능력을 가지고 있어요.",
            "명대사": "We're going to get through this"
        }
    },
    "브리저튼": {
        "다프네 브리저튼": {
            "성격": "독립적이고 똑똑한 여성으로, 사회의 관습에 맞서 자신의 행복을 찾는 강인한 인물이에요!",
            "특징": "독서를 좋아하고 지적 호기심이 많으며, 진정한 사랑을 믿는 낭만적인 면도 있어요.",
            "명대사": "I will not be defined by my marriage"
        }
    },
    "위쳐": {
        "게롤트": {
            "성격": "냉정하고 무뚝뚝해 보이지만, 사실은 따뜻한 마음을 가진 마법사 사냥꾼이에요!",
            "특징": "강력한 전투력을 가지고 있으며, 정의를 위해 싸우지만 감정 표현은 서툴러요.",
            "명대사": "Hmm"
        },
        "시리": {
            "성격": "강력한 마법 능력을 가진 공주로, 게롤트의 양녀이에요. 똑똑하고 용감하며, 자신의 운명을 스스로 결정하려는 의지가 강해요!",
            "특징": "고대 혈통의 힘을 가지고 있으며, 위험한 상황에서도 침착하게 대처하는 능력을 가지고 있어요.",
            "명대사": "I'm not a child anymore"
        }
    },
    "루시퍼": {
        "루시퍼 모닝스타": {
            "성격": "지옥의 왕이지만 인간 세계에서 나이트클럽을 운영하며 살아가는 매력적이고 유머러스한 캐릭터예요!",
            "특징": "매우 매력적이고 카리스마가 있으며, 솔직하고 직설적인 성격이에요.",
            "명대사": "What is it you truly desire?"
        }
    },
    "스트레인저 씽스": {
        "조이스 바이어스": {
            "성격": "아들을 찾기 위해 모든 것을 감수하는 강인한 어머니예요! 용감하고 결단력이 있으며, 포기하지 않는 의지가 있어요.",
            "특징": "어머니의 사랑이 얼마나 강한지 보여주는 캐릭터로, 어떤 어려움도 극복하려는 의지가 있어요.",
            "명대사": "I'm going to find my son"
        }
    }
}

def get_azure_openai_client():
    """Azure OpenAI 클라이언트 생성"""
    if not azure_oai_endpoint or not azure_oai_key:
        return None
    
    try:
        return AzureOpenAI(
            azure_endpoint=azure_oai_endpoint,
            api_key=azure_oai_key,
            api_version=azure_oai_api_version
        )
    except Exception as e:
        st.error(f"Azure OpenAI 클라이언트 생성 오류: {str(e)}")
        return None

def get_character_info(series_name, character_name=None):
    """시리즈와 캐릭터 정보 가져오기"""
    if series_name not in NETFLIX_CHARACTERS:
        return None
    
    characters = NETFLIX_CHARACTERS[series_name]
    
    if character_name:
        return characters.get(character_name)
    else:
        return characters

def format_character_response(series_name, character_name=None):
    """캐릭터 정보를 귀엽고 깜찍한 말투로 포맷팅"""
    characters = get_character_info(series_name, character_name)
    
    if not characters:
        return f"어라라~ 😅 {series_name} 시리즈의 정보를 찾을 수 없어요! 다른 넷플릭스 작품을 물어봐주세요~"
    
    response = f"🎬 **{series_name}** 캐릭터 정보예요!\n\n"
    
    if character_name and character_name in characters:
        # 특정 캐릭터 정보
        char = characters[character_name]
        response += f"## ✨ {character_name} ✨\n\n"
        response += f"### 🎭 성격\n{char['성격']}\n\n"
        response += f"### 🌟 특징\n{char['특징']}\n\n"
        response += f"### 💬 명대사\n\"{char['명대사']}\"\n\n"
        response += "---\n\n"
        response += "이 캐릭터에 대해 더 궁금한 게 있으면 언제든 물어봐주세요! 😊"
    else:
        # 모든 캐릭터 정보
        response += "이 작품의 주요 캐릭터들이에요:\n\n"
        for name, info in characters.items():
            response += f"### 🎪 {name}\n"
            response += f"**성격**: {info['성격']}\n\n"
            response += f"**특징**: {info['특징']}\n\n"
            response += f"**명대사**: \"{info['명대사']}\"\n\n"
            response += "---\n\n"
    
    return response

def get_chat_response(user_message, conversation_history):
    """Azure OpenAI를 사용하여 귀엽고 깜찍한 말투로 챗봇 응답 생성"""
    client = get_azure_openai_client()
    
    if not client:
        return "⚠️ 어라라~ Azure OpenAI 설정이 완료되지 않았어요! .env 파일을 확인해주세요~ 😅"
    
    # 귀엽고 깜찍한 말투의 시스템 프롬프트
    system_prompt = """당신은 넷플릭스 드라마와 영화의 캐릭터 성격을 알려주는 귀엽고 깜찍한 챗봇이에요!

말투 규칙:
- 항상 "~해요", "~예요", "~어요" 같은 존댓말을 사용해요
- 이모지를 적절히 사용해서 친근하게 대답해요 (예: 😊, 🎬, ✨, 💕, 🎭)
- "어라라~", "와아~", "헤헤~" 같은 귀여운 감탄사를 사용해요
- 매우 친근하고 따뜻한 톤으로 대답해요
- "궁금한 게 있으면 언제든 물어봐주세요!" 같은 친절한 표현을 사용해요

사용자가 넷플릭스 작품의 캐릭터에 대해 물어보면:
- 해당 캐릭터의 성격, 특징, 명대사 등을 귀엽고 친근하게 설명해주세요
- 작품의 배경과 스토리도 간단히 언급해주세요
- 캐릭터의 매력 포인트를 강조해서 설명해주세요

한국어로 자연스럽고 귀엽게 대답해주세요!"""
    
    # 대화 기록 구성
    messages = [{"role": "system", "content": system_prompt}]
    
    # 이전 대화 기록 추가 (최근 10개만 유지)
    for msg in conversation_history[-10:]:
        messages.append(msg)
    
    # 현재 사용자 메시지 추가
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model=azure_oai_deployment,
            messages=messages,
            temperature=0.8,  # 더 창의적이고 귀여운 답변을 위해 온도 상승
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"어라라~ 오류가 발생했어요! 😅 {str(e)}\n\n다시 시도해주세요~"

def extract_series_and_character(user_message):
    """사용자 메시지에서 시리즈명과 캐릭터명 추출"""
    series_list = list(NETFLIX_CHARACTERS.keys())
    
    found_series = None
    found_character = None
    
    for series in series_list:
        if series in user_message:
            found_series = series
            characters = NETFLIX_CHARACTERS.get(series, {})
            for char_name in characters.keys():
                if char_name in user_message:
                    found_character = char_name
                    break
            break
    
    return found_series, found_character

def main():
    st.set_page_config(
        page_title="넷플릭스 캐릭터 성격 챗봇",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 커스텀 CSS로 귀여운 디자인 적용
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E50914;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎬 넷플릭스 캐릭터 성격 챗봇 🎭</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">귀엽고 깜찍한 챗봇이 넷플릭스 캐릭터들의 성격을 알려드려요! 💕</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # 환영 메시지 추가
        welcome_msg = "안녕하세요! 🎉 넷플릭스 캐릭터 성격 챗봇이에요~ 😊\n\n어떤 넷플릭스 작품의 캐릭터가 궁금하신가요? 예를 들어:\n- '오징어 게임 성기훈 성격 알려줘'\n- '킹덤 이창은 어떤 사람이야?'\n- '기묘한 이야기 일레븐에 대해 알려줘'\n\n언제든 물어보세요! 💕"
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # 사이드바 - 빠른 조회
    with st.sidebar:
        st.header("🎬 빠른 조회")
        st.markdown("---")
        
        selected_series = st.selectbox(
            "작품 선택",
            ["작품을 선택하세요"] + list(NETFLIX_CHARACTERS.keys())
        )
        
        if selected_series != "작품을 선택하세요":
            characters = NETFLIX_CHARACTERS.get(selected_series, {})
            if characters:
                selected_character = st.selectbox(
                    "캐릭터 선택",
                    ["전체"] + list(characters.keys())
                )
                
                if st.button("✨ 정보 조회", use_container_width=True):
                    if selected_character == "전체":
                        info = format_character_response(selected_series)
                    else:
                        info = format_character_response(selected_series, selected_character)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": info
                    })
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### 📺 지원 작품 목록")
        for series in NETFLIX_CHARACTERS.keys():
            st.markdown(f"- 🎬 {series}")
        
        st.markdown("---")
        st.markdown("### 💡 사용 팁")
        st.info("""
        💬 자연스럽게 대화하듯 물어보세요!
        
        예시:
        - "오징어 게임 캐릭터들 알려줘"
        - "성기훈은 어떤 사람이야?"
        - "킹덤 이창 성격이 궁금해"
        """)
    
    # 채팅 인터페이스
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("넷플릭스 캐릭터에 대해 물어보세요! (예: 오징어 게임 성기훈 성격 알려줘)"):
        # 사용자 메시지 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 시리즈와 캐릭터 추출
        series, character = extract_series_and_character(prompt)
        
        # 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("캐릭터 정보를 찾고 있어요... 🎬✨"):
                if series:
                    # 캐릭터 정보가 있는 경우 직접 제공
                    character_info = format_character_response(series, character)
                    
                    # Azure OpenAI로 추가 설명 생성
                    ai_response = get_chat_response(
                        f"사용자가 {series}의 {character if character else '캐릭터들'}에 대해 물어봤어요. 다음 정보를 바탕으로 귀엽고 친근하게 추가 설명을 해주세요:\n\n{character_info}",
                        st.session_state.conversation_history
                    )
                    
                    response = f"{character_info}\n\n---\n\n{ai_response}"
                else:
                    # 일반적인 질문은 Azure OpenAI로 처리
                    response = get_chat_response(prompt, st.session_state.conversation_history)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.conversation_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

